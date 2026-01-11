import polars.exceptions
import polars as pl


class KaplanMeier:
    def __init__(self):
        self.df = None
        self.result = None


    def fit(self, df):
        needed_columns = ["time", "censored"]
        try:
            for variable in needed_columns:
                df[variable]
        except polars.exceptions.ColumnNotFoundError as e:
            print(f"Necessary column {variable} not found in df")
            return
        self.df = df
        self.__fit()
        return self.result

    def __fit(self):
        self.result = (
            (
                self.df
                # Bereken de m (censor = 1) en q (censor = 0) aantallen voor elk tijds-punt
                .group_by("time")
                .agg(
                    m=pl.col("censored").sum().cast(pl.Int64),
                    q=pl.col("censored").eq(0).sum().cast(pl.Int64)
                ).sort("time")

                # Geef de censor van i+1 aan de q van i als m[i+]
                .with_columns(
                    next_q=pl.when(pl.col("m").shift(1) > 0)
                    .then(pl.col("q").shift(-1))
                    .otherwise(0)
                )
                .with_columns(
                    q=pl.when(pl.col("next_q").is_null() | pl.col("next_q") == 0)
                    .then(pl.col("q"))
                    .otherwise(pl.col("next_q"))
                )
                # Haal alle tijden met m < 0 weg.
                .filter(pl.col("m") > 0).drop("next_q")
                .vstack(pl.DataFrame({"time": [0], "m": [0], "q": [0]}))
                .sort("time")

                # Bereken de afname van subjects overtijd.
                .with_columns(
                    total_subjects=pl.lit(len(self.df)),
                    cumulative_m=pl.col("m").cum_sum(),
                    cumulative_q=pl.col("q").cum_sum()
                )
                .with_columns(
                    n=pl.col("total_subjects") - pl.col("cumulative_m").shift(1) - pl.col("cumulative_q").shift(1)
                )
                .with_columns(
                    n=pl.when(pl.col("time") == 0).then(pl.col("total_subjects")).otherwise(pl.col("n"))
                ).select(["time", "m", "q", "n"])
            )
        )
        self.__calc_survival_probability()

    def __calc_survival_probability(self):
        self.result = (
            self.result
            .with_columns(
                prob_survive=(pl.col("n") - pl.col("m")) / pl.col("n")
            ).with_columns(
                # Cumulative product starting from 1.0
                s=pl.col("prob_survive").cum_prod()
            ).drop("prob_survive")
        )

