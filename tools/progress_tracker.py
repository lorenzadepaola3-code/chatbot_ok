from collections import defaultdict

TARGET = {
    "inflation_dynamics": 25,
    "core_vs_headline": 20,
    "policy_stance": 30,
    "balance_sheet": 30,
    "tltro": 20,
    "wages_labour": 25,
    "digital_euro": 35,
    "payments_innovation": 20,
    "financial_stability": 25,
    "climate": 20,
    "energy_shocks": 25,
    "forward_guidance": 25,
}


def compare(current_counts):
    report = []
    for k, v in TARGET.items():
        cur = current_counts.get(k, 0)
        pct = cur / v * 100
        report.append(f"{k:20s} {cur:3d}/{v:3d} ({pct:5.1f}%)")
    return "\n".join(report)


if __name__ == "__main__":
    # plug in actual counts from validate_dataset.stats()
    dummy = defaultdict(int, financial_stability=2)
    print(compare(dummy))
