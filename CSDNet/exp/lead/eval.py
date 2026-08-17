#!/usr/bin/env python
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-d", "--sim_thr", type=float, default=0.4)
    parser.add_argument("--planned_total", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.file, names=["smiles", "DS", "QED", "SA", "SIM", ""])
    raw_n = len(df)
    num_gen = args.planned_total if args.planned_total > 0 else raw_n
    df = df.drop_duplicates(subset=["smiles"])
    print(f"Generated:\t{raw_n}/{num_gen}")
    if raw_n:
        print(f"Uniqueness(actual):\t{len(df) / raw_n}")
    print(f"Unique/planned:\t{len(df) / num_gen}")

    df = df[df["SIM"] >= args.sim_thr]
    df = df[df["QED"] >= 0.6]
    df = df[df["SA"] >= 6 / 9]
    if not len(df):
        print("Lead optimization failed")
        return

    df = df.sort_values(by="DS", ascending=False)
    print(f"Top DS:\t\t{df['DS'].iloc[0]}")
    print(f"Top mol:\t{df['smiles'].iloc[0]}")


if __name__ == "__main__":
    main()
