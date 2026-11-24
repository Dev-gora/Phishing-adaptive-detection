"""URL-only feature extraction for live analysis.

The benchmark stores 87 features, but only the URL-derived ones can be computed
from a URL string alone. Content features need the page to be fetched, and
external features need third-party services (search index, page rank, WHOIS).
Rather than invent those values, this module computes only the string-derived
subset and the application trains a dedicated model on exactly those features.

Every feature name here matches a column of the published dataset, so the same
definition used at training time is used at serving time. ``fidelity_report``
measures how often this extractor reproduces the published column values on a
sample of the dataset, which is the honest way to bound the train/serve skew.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List
from urllib.parse import urlsplit

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "shorte.st", "bl.ink", "lnkd.in", "db.tt", "qr.ae", "cur.lv",
    "ity.im", "q.gs", "po.st", "bc.vc", "twitthis.com", "u.to", "j.mp",
    "buzurl.com", "cutt.us", "u.bb", "yourls.org", "x.co", "prettylinkpro.com",
    "scrnch.me", "filoops.info", "vzturl.com", "qr.net", "1url.com", "tweez.me",
    "v.gd", "tr.im", "link.zip", "s.id", "surl.li", "short.gy", "clck.ru",
}

PHISH_HINTS = {
    "login", "signin", "sign-in", "secure", "account", "update", "verify",
    "verification", "confirm", "password", "credential", "bank", "banking",
    "paypal", "ebay", "amazon", "apple", "microsoft", "office365", "outlook",
    "webscr", "wp-admin", "admin", "support", "billing", "invoice", "payment",
    "recover", "unlock", "alert", "notification", "authenticate", "authorize",
    "session", "token", "wallet", "crypto", "bonus", "free", "gift", "prize",
    "winner", "claim", "urgent", "suspended", "limited", "reset",
}

SUSPICIOUS_TLDS = {
    "zip", "mov", "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work",
    "loan", "click", "link", "fit", "rest", "cam", "surf", "icu", "buzz",
    "monster", "quest", "cyou", "sbs", "cfd", "bargains", "date", "review",
    "stream", "download", "win", "bid", "trade", "webcam", "party", "racing",
}

RISKY_EXTENSIONS = {
    "exe", "scr", "bat", "cmd", "com", "pif", "vb", "vbs", "js", "jse",
    "jar", "ps1", "msi", "hta", "cpl", "dll", "apk", "bin", "zip", "rar",
    "7z", "gz", "tar", "php", "asp", "aspx", "jsp", "cgi", "pl",
}


def _split(url: str):
    raw = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    parts = urlsplit(raw)
    host = parts.netloc.split("@")[-1]
    port = ""
    if ":" in host:
        host, _, port = host.partition(":")
    return raw, parts, host.lower(), port, parts.path or "", parts.query or ""


def _tokens(value: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", value) if t]


def _max_char_repeat(value: str) -> int:
    best = 0
    run = 0
    previous = ""
    for ch in value:
        if ch == previous:
            run += 1
        else:
            run = 1
            previous = ch
        best = max(best, run)
    return 0 if best == 1 else best


def _tld_of(host: str) -> str:
    labels = [x for x in host.split(".") if x]
    return labels[-1] if labels else ""


def extract(url: str) -> Dict[str, float]:
    """Compute the URL-derived feature vector for one URL.

    Values are floats to match the dataset's numeric columns exactly.
    """
    raw, parts, host, port, path, query = _split(url)
    labels = [x for x in host.split(".") if x]
    subdomain = ".".join(labels[:-2]) if len(labels) > 2 else ""
    tld = _tld_of(host)
    scheme = parts.scheme.lower()

    url_digits = sum(c.isdigit() for c in raw)
    host_digits = sum(c.isdigit() for c in host)

    raw_tokens = _tokens(raw)
    host_tokens = _tokens(host)
    path_tokens = _tokens(path)

    def stats(tokens: List[str]):
        if not tokens:
            return 0.0, 0.0, 0.0
        lengths = [len(t) for t in tokens]
        return float(min(lengths)), float(max(lengths)), float(sum(lengths) / len(lengths))

    shortest_raw, longest_raw, avg_raw = stats(raw_tokens)
    shortest_host, longest_host, avg_host = stats(host_tokens)
    shortest_path, longest_path, avg_path = stats(path_tokens)

    hints = 0
    lowered = raw.lower()
    for word in PHISH_HINTS:
        if word in lowered:
            hints += 1

    path_lower = path.lower()
    tld_in_path = 1.0 if any(path_lower.endswith("." + ext) for ext in ("com", "net", "org", "edu", "gov", "info", "biz")) or re.search(r"\.[a-z]{2,4}/", path_lower + "/") else 0.0
    tld_in_subdomain = 1.0 if subdomain and any(lbl in {"com", "net", "org", "edu", "gov", "info", "biz"} for lbl in subdomain.split(".")) else 0.0

    return {
        "length_url": float(len(raw)),
        "length_hostname": float(len(host)),
        "ip": 1.0 if IP_RE.match(host) else 0.0,
        "nb_dots": float(raw.count(".")),
        "nb_hyphens": float(raw.count("-")),
        "nb_at": float(raw.count("@")),
        "nb_qm": float(raw.count("?")),
        "nb_and": float(raw.count("&")),
        "nb_eq": float(raw.count("=")),
        "nb_underscore": float(raw.count("_")),
        "nb_tilde": float(raw.count("~")),
        "nb_percent": float(raw.count("%")),
        "nb_slash": float(raw.count("/")),
        "nb_star": float(raw.count("*")),
        "nb_colon": float(raw.count(":")),
        "nb_comma": float(raw.count(",")),
        "nb_semicolumn": float(raw.count(";")),
        "nb_dollar": float(raw.count("$")),
        "nb_space": float(raw.count(" ") + raw.count("%20")),
        "nb_www": float(raw.count("www")),
        "nb_com": float(raw.count(".com")),
        "nb_dslash": 1.0 if "//" in path else 0.0,
        "http_in_path": float(path.lower().count("http")),
        "https_token": 0.0 if "https" in raw.lower() else 1.0,
        "ratio_digits_url": url_digits / len(raw) if raw else 0.0,
        "ratio_digits_host": host_digits / len(host) if host else 0.0,
        "punycode": 1.0 if "xn--" in host else 0.0,
        "port": 1.0 if port else 0.0,
        "tld_in_path": tld_in_path,
        "tld_in_subdomain": tld_in_subdomain,
        "abnormal_subdomain": 1.0 if subdomain and (subdomain.count(".") >= 2 or re.search(r"w{2,}", subdomain)) else 0.0,
        "nb_subdomains": float(max(0, len(labels) - 2)),
        "prefix_suffix": 1.0 if "-" in host else 0.0,
        "shortening_service": 1.0 if host in SHORTENERS else 0.0,
        "path_extension": 1.0 if ("." in path.rsplit("/", 1)[-1] and path.rsplit(".", 1)[-1].lower() in RISKY_EXTENSIONS) else 0.0,
        "length_words_raw": float(len(raw_tokens)),
        "char_repeat": float(_max_char_repeat(raw)),
        "shortest_words_raw": shortest_raw,
        "shortest_word_host": shortest_host,
        "shortest_word_path": shortest_path,
        "longest_words_raw": longest_raw,
        "longest_word_host": longest_host,
        "longest_word_path": longest_path,
        "avg_words_raw": avg_raw,
        "avg_word_host": avg_host,
        "avg_word_path": avg_path,
        "phish_hints": float(hints),
        "suspecious_tld": 1.0 if tld in SUSPICIOUS_TLDS else 0.0,
    }


# Human-readable descriptions used by the feature dictionary in the UI.
DESCRIPTIONS: Dict[str, str] = {
    "length_url": "Total number of characters in the URL, including the scheme.",
    "length_hostname": "Number of characters in the hostname (domain plus subdomains).",
    "ip": "1 when the host is a raw IPv4 address instead of a domain name.",
    "nb_dots": "Number of '.' characters in the whole URL.",
    "nb_hyphens": "Number of '-' characters in the whole URL.",
    "nb_at": "Number of '@' characters, which can hide the true host in a URL.",
    "nb_qm": "Number of '?' characters (query-string separators).",
    "nb_and": "Number of '&' characters (parameter separators).",
    "nb_eq": "Number of '=' characters (parameter assignments).",
    "nb_underscore": "Number of '_' characters.",
    "nb_tilde": "Number of '~' characters.",
    "nb_percent": "Number of '%' characters, used in percent-encoding.",
    "nb_slash": "Number of '/' characters, including the two in the scheme separator.",
    "nb_star": "Number of '*' characters.",
    "nb_colon": "Number of ':' characters, including a port separator.",
    "nb_comma": "Number of ',' characters.",
    "nb_semicolumn": "Number of ';' characters.",
    "nb_dollar": "Number of '$' characters.",
    "nb_space": "Number of spaces or %20 sequences in the URL.",
    "nb_www": "Occurrences of the string 'www'.",
    "nb_com": "Occurrences of the string '.com' anywhere in the URL.",
    "nb_dslash": "1 when a double slash appears inside the path rather than only after the scheme.",
    "http_in_path": "Occurrences of the string 'http' inside the path.",
    "https_token": "1 when the string 'https' appears anywhere in the URL.",
    "ratio_digits_url": "Digits in the URL divided by the URL length.",
    "ratio_digits_host": "Digits in the hostname divided by the hostname length.",
    "punycode": "1 when the host uses punycode ('xn--'), often used to imitate a brand.",
    "port": "1 when an explicit port number is present.",
    "tld_in_path": "1 when a top-level domain token appears inside the path.",
    "tld_in_subdomain": "1 when a top-level domain token appears in a subdomain label.",
    "abnormal_subdomain": "1 when the subdomain is unusually deep or repeats the letter 'w'.",
    "nb_subdomains": "Number of labels in the hostname beyond the registrable domain.",
    "prefix_suffix": "1 when the hostname contains a hyphen, the classic brand-prefix pattern.",
    "shortening_service": "1 when the host belongs to a known URL-shortening service.",
    "path_extension": "1 when the final path segment ends in an executable or script extension.",
    "length_words_raw": "Number of alphanumeric tokens in the whole URL.",
    "char_repeat": "Longest run of the same character repeated consecutively.",
    "shortest_words_raw": "Length of the shortest token in the whole URL.",
    "shortest_word_host": "Length of the shortest token in the hostname.",
    "shortest_word_path": "Length of the shortest token in the path.",
    "longest_words_raw": "Length of the longest token in the whole URL.",
    "longest_word_host": "Length of the longest token in the hostname.",
    "longest_word_path": "Length of the longest token in the path.",
    "avg_words_raw": "Mean token length across the whole URL.",
    "avg_word_host": "Mean token length across the hostname.",
    "avg_word_path": "Mean token length across the path.",
    "phish_hints": "Number of suspicious keywords found (login, secure, verify, account, ...).",
    "suspecious_tld": "1 when the top-level domain is one frequently abused for phishing.",
}


def feature_order() -> List[str]:
    return list(DESCRIPTIONS)


def fidelity_report(df, sample_n: int = 400, seed: int = 0) -> Dict[str, dict]:
    """Compare this extractor with the published columns on a sample of rows.

    Returns per-feature agreement: exact-match rate for discrete columns and
    mean absolute error / correlation for continuous ones.
    """
    import numpy as np

    available = [f for f in DESCRIPTIONS if f in df.columns]
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed)
    computed = [extract(u) for u in sample["url"].tolist()]

    report: Dict[str, dict] = {}
    for feature in available:
        truth = sample[feature].to_numpy(dtype=float)
        pred = np.array([c[feature] for c in computed], dtype=float)
        exact = float(np.mean(np.isclose(pred, truth)))
        mae = float(np.mean(np.abs(pred - truth)))
        if np.std(pred) > 1e-9 and np.std(truth) > 1e-9:
            corr = float(np.corrcoef(pred, truth)[0, 1])
        else:
            corr = 1.0 if np.allclose(pred, truth) else 0.0
        report[feature] = {
            "exact_match": round(exact, 4),
            "mae": round(mae, 4),
            "correlation": round(corr, 4),
            "discrete": bool(sample[feature].nunique() <= 6),
        }
    return report


def select_reliable(report: Dict[str, dict], exact_threshold: float = 0.95,
                    corr_threshold: float = 0.95) -> List[str]:
    """Keep features this extractor reproduces faithfully enough to serve."""
    keep = []
    for feature, stats in report.items():
        if stats["discrete"]:
            if stats["exact_match"] >= exact_threshold:
                keep.append(feature)
        elif stats["correlation"] >= corr_threshold and stats["mae"] <= 2.5:
            keep.append(feature)
    return sorted(keep)
