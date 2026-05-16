from pyspark.sql import functions as F


def profile_dataframe(df):

    # Cache dataframe
    df.cache()
    df.count()

    print("\n" + "=" * 70)
    print("TABLE 1 — COLUMN PROFILE")
    print("=" * 70)

    # Null counts in ONE query
    null_exprs = [
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]

    null_counts = df.select(null_exprs).collect()[0].asDict()

    profile_rows = []

    for col_name, dtype in df.dtypes:

        unique_count = df.select(col_name).distinct().count()

        profile_rows.append((
            col_name,
            dtype,
            null_counts[col_name],
            unique_count
        ))

    for row in profile_rows:
        print(row)

    print("\n" + "=" * 70)
    print("TABLE 2 — INTEGER ANALYSIS")
    print("=" * 70)

    int_cols = [
        col_name
        for col_name, dtype in df.dtypes
        if dtype in ["int", "bigint","double"]
    ]

    stats_exprs = []

    for c in int_cols:

        stats_exprs.extend([
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.avg(c).alias(f"{c}_avg")
        ])

    stats = df.select(stats_exprs).collect()[0].asDict()

    for c in int_cols:

        print(
            c,
            stats[f"{c}_min"],
            stats[f"{c}_max"],
            round(stats[f"{c}_avg"], 2)
        )

    print("\n" + "=" * 70)
    print("TABLE 3 — LOW CARDINALITY")
    print("=" * 70)

    for c in df.columns:

        unique_count = df.select(c).distinct().count()

        if unique_count < 8:

            values = [
                row[0]
                for row in df.select(c)
                .distinct()
                .collect()
            ]

            print(f"{c:<25} {values}")

    df.unpersist()