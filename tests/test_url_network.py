from datetime import datetime, timezone
from types import SimpleNamespace

import httpx

from app.services.cache import SqliteCache
from app.services.url_network import (
    domain_age_days,
    expand_short_url,
    is_public_host,
)

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


def resolver_for(mapping):
    def resolve(host, port):
        if host not in mapping:
            raise OSError("no such host")
        return [(2, 1, 6, "", (mapping[host], 0))]

    return resolve


def test_private_and_loopback_addresses_are_not_public():
    for address in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "0.0.0.0", "::1"):
        assert not is_public_host("h", resolver_for({"h": address})), address
    assert is_public_host("h", resolver_for({"h": "93.184.216.34"}))
    assert not is_public_host("unresolvable", resolver_for({}))


def test_expand_short_url_follows_redirects_to_the_final_url():
    hops = {
        "https://bit.ly/x": "https://mid.example/y",
        "https://mid.example/y": "https://final.example/z",
        "https://final.example/z": None,
    }

    def request(url, **kwargs):
        assert kwargs["follow_redirects"] is False
        target = hops[url]
        return SimpleNamespace(status_code=302 if target else 200, headers={"location": target} if target else {})

    resolver = resolver_for({"bit.ly": "1.1.1.1", "mid.example": "2.2.2.2", "final.example": "3.3.3.3"})
    assert expand_short_url("https://bit.ly/x", request=request, resolver=resolver) == "https://final.example/z"


def test_expand_short_url_refuses_a_redirect_into_a_private_network():
    def request(url, **kwargs):
        return SimpleNamespace(status_code=302, headers={"location": "http://internal.example/admin"})

    resolver = resolver_for({"bit.ly": "1.1.1.1", "internal.example": "10.0.0.8"})
    assert expand_short_url("https://bit.ly/x", request=request, resolver=resolver) is None


def test_expand_short_url_refuses_odd_ports_schemes_and_loops():
    resolver = resolver_for({"bit.ly": "1.1.1.1"})
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not request"))
    assert expand_short_url("https://bit.ly:8080/x", request=never, resolver=resolver) is None
    assert expand_short_url("ftp://bit.ly/x", request=never, resolver=resolver) is None

    loop = lambda url, **k: SimpleNamespace(status_code=302, headers={"location": "https://bit.ly/x"})
    assert expand_short_url("https://bit.ly/x", request=loop, resolver=resolver, max_hops=3) is None


def test_expand_short_url_returns_none_on_network_error():
    def request(url, **kwargs):
        raise httpx.ConnectError("down")

    assert expand_short_url("https://bit.ly/x", request=request, resolver=resolver_for({"bit.ly": "1.1.1.1"})) is None


class FakeRdap:
    def __init__(self, registered="2026-10-20T00:00:00Z", status=200):
        self.calls = 0
        self.registered = registered
        self.status = status

    def __call__(self, url, **kwargs):
        self.calls += 1
        if "iana.org" in url:
            payload = {"services": [[["com", "top"], ["https://rdap.example/"]]]}
            return SimpleNamespace(status_code=200, json=lambda: payload, raise_for_status=lambda: None)
        payload = {"events": [{"eventAction": "registration", "eventDate": self.registered}]}
        return SimpleNamespace(status_code=self.status, json=lambda: payload)


def test_domain_age_is_computed_from_the_registration_event():
    assert domain_age_days("fresh.top", get=FakeRdap(), now=NOW) == 8


def test_domain_age_is_cached_between_calls():
    cache, get = SqliteCache(":memory:"), FakeRdap()
    assert domain_age_days("fresh.top", cache=cache, get=get, now=NOW) == 8
    calls_after_first = get.calls
    assert domain_age_days("fresh.top", cache=cache, get=get, now=NOW) == 8
    assert get.calls == calls_after_first


def test_unregistered_domain_has_no_age_and_transient_errors_are_not_cached():
    assert domain_age_days("nothere.top", get=FakeRdap(status=404), now=NOW) is None
    cache, flaky = SqliteCache(":memory:"), FakeRdap(status=500)
    assert domain_age_days("x.top", cache=cache, get=flaky, now=NOW) is None
    assert cache.get("domain_age", "x.top") is None


def test_domain_age_ignores_ip_literals_and_bare_names():
    def never(*args, **kwargs):
        raise AssertionError("must not query RDAP")

    for value in ("203.0.113.9", "localhost", ""):
        assert domain_age_days(value, get=never) is None
