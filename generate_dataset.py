"""Generate a synthetic phishing website dataset.

The data is fictional, but feature distributions are designed to resemble
realistic differences between legitimate and phishing pages.
"""
from __future__ import annotations

import ipaddress
import random
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BRANDS = [
    "sber", "tbank", "vk", "yandex", "gosuslugi", "mirea", "ozon", "wildberries",
    "steam", "paypal", "google", "apple", "telegram", "mail", "avito"
]
SAFE_DOMAINS = [
    "mirea.ru", "yandex.ru", "vk.com", "ozon.ru", "tinkoff.ru", "sberbank.ru",
    "gosuslugi.ru", "apple.com", "google.com", "telegram.org", "avito.ru",
    "wildberries.ru", "steamcommunity.com", "mail.ru"
]
SUSPICIOUS_WORDS = [
    "login", "verify", "bonus", "gift", "free", "secure", "account", "update",
    "confirm", "wallet", "payment", "prize", "support", "auth"
]
TLD_BAD = [".ru.com", ".top", ".xyz", ".click", ".shop", ".icu", ".live", ".site"]


def random_ip() -> str:
    return str(ipaddress.IPv4Address(random.randint(0x0B000001, 0xDF0000FF)))


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def build_legit_url() -> str:
    domain = random.choice(SAFE_DOMAINS)
    sub = random.choice(["", "www.", "help.", "my.", "id."])
    paths = ["", "/", "/profile", "/news", "/support", "/catalog", "/login", "/education"]
    query = random.choice(["", "", "", f"?ref={random.randint(10,99)}"])
    return f"https://{sub}{domain}{random.choice(paths)}{query}"


def build_phishing_url() -> str:
    brand = random.choice(BRANDS)
    word = random.choice(SUSPICIOUS_WORDS)
    sep = random.choice(["-", "_", "", "."])
    token = random.choice([str(random.randint(1000, 999999)), "auth", "id", "secure", "pay"])
    tld = random.choice(TLD_BAD)

    variant = random.random()
    if variant < 0.18:
        host = random_ip()
    elif variant < 0.55:
        host = f"{brand}{sep}{word}{sep}{token}{tld}"
    else:
        # Brand appears in subdomain, while the real registered domain is unrelated.
        host = f"{brand}.{word}-{token}{tld}"

    scheme = random.choice(["http", "http", "https"])
    path = random.choice([
        "/login", "/verify/account", "/bonus", "/payment/confirm",
        f"/{brand}/signin", "/security-check", "/update"
    ])
    if random.random() < 0.55:
        path += f"?session={random.randint(100000, 999999999)}&u={random.choice(BRANDS)}"
    if random.random() < 0.10:
        host = f"user@{host}"
    return f"{scheme}://{host}{path}"


def extract_url_features(url: str) -> dict[str, int | float]:
    lower = url.lower()
    after_scheme = lower.split("://", 1)[-1]
    host = after_scheme.split("/", 1)[0]
    host_no_auth = host.split("@")[-1]

    has_ip = 0
    try:
        ipaddress.ip_address(host_no_auth)
        has_ip = 1
    except ValueError:
        pass

    brand_in_subdomain = 0
    parts = host_no_auth.split(".")
    if len(parts) >= 3:
        subdomain = ".".join(parts[:-2])
        brand_in_subdomain = int(any(brand in subdomain for brand in BRANDS))

    return {
        "url_length": len(url),
        "num_dots": lower.count("."),
        "num_hyphens": lower.count("-"),
        "num_digits": count_digits(url),
        "has_ip": has_ip,
        "has_at_symbol": int("@" in host),
        "uses_https": int(lower.startswith("https://")),
        "suspicious_words_count": sum(word in lower for word in SUSPICIOUS_WORDS),
        "brand_in_subdomain": brand_in_subdomain,
    }


def make_row(label: int) -> dict[str, int | float | str]:
    url = build_phishing_url() if label else build_legit_url()
    row = extract_url_features(url)

    if label:
        row.update({
            "domain_age_days": max(1, int(random.gauss(260, 260))),
            "external_links_ratio": min(1, max(0, random.gauss(0.60, 0.22))),
            "forms_count": max(0, int(random.gauss(1.8, 1.3))),
            "popup_count": max(0, int(random.gauss(1.1, 1.0))),
            "ssl_valid": int(random.random() < 0.58),
            "redirect_count": max(0, int(random.gauss(1.6, 1.2))),
        })
    else:
        row.update({
            "domain_age_days": max(20, int(random.gauss(1600, 1200))),
            "external_links_ratio": min(1, max(0, random.gauss(0.32, 0.18))),
            "forms_count": max(0, int(random.gauss(1.1, 1.0))),
            "popup_count": max(0, int(random.gauss(0.35, 0.7))),
            "ssl_valid": int(random.random() < 0.88),
            "redirect_count": max(0, int(random.gauss(0.7, 0.85))),
        })

    # Add noise: not every phishing/legit case is obvious.
    if random.random() < 0.14:
        row["uses_https"] = 1 - int(row["uses_https"])
    if random.random() < 0.12:
        row["ssl_valid"] = 1 - int(row["ssl_valid"])

    row["url"] = url
    row["label"] = label
    return row


def generate_dataset(n_rows: int = 1500) -> pd.DataFrame:
    rows = []
    for _ in range(n_rows):
        label = int(random.random() < 0.48)
        row = make_row(label)
        # Small label noise makes the task closer to reality: some phishing pages
        # look almost legitimate and some legitimate pages have risky patterns.
        if random.random() < 0.045:
            row["label"] = 1 - int(row["label"])
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["url"] + [c for c in df.columns if c not in {"url", "label"}] + ["label"]
    return df[cols]


if __name__ == "__main__":
    out = Path("data/phishing_sites_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.head())
