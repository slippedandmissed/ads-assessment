from typing import TypedDict, Optional, Dict
from datetime import date as date_t
import statsmodels.api as sm
from . import access, assess
import numpy as np
import dateutil


class Prediction(TypedDict):
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence: float


default_tags: Dict[str, bool] = {
    "cuisine": True,
    "shop": True,
    "tourism": True,
    "healthcare": True
}


def predict_price(
    lat: float,
    lng: float,
    date: date_t,
    property_type: str,
    train_prop: float = 0.8,
    train_set_radius_km: float = 10,
    train_set_time_period_days: int = 3,
    poi_radius_km: float = 0.5,
    seed: Optional[int] = None,
    tags: Dict[str, bool] = default_tags,
    verbose: bool = True,
) -> Prediction:
    """Predict the price at which a property sold."""

    # This box will contain all of the properties in the training and validation sets
    property_box = assess.calculate_box(
        lat,
        lng,
        train_set_radius_km,
        date,
        train_set_time_period_days
    )

    df = assess.data(
        property_box,
        tags,
        poi_radius_km,
        verbose
    )

    labelled = assess.labelled(df, tags, seed=seed)

    # Make sure there's enough data to train from
    if len(labelled["x_train"]) < 100:
        print(
            f"Warning: There are only {len(labelled['x_train'])} datapoints in the training set. The model will likely be poor. Try increasing train_set_radius_km or train_set_time_period_days.")

    # I've chosen the Gamma family because I'm expecting
    # the prices to be heavily skewed. Specifically, I
    # expect that there will be far more lower-priced
    # properties than higher priced ones.
    #
    # I've also chosen a log link function because from
    # observation of the data, the relationships between
    # the indicator variables and the log price seems to
    # be a lot closer to linear than those of the raw price.
    #
    # Note that the model is only fitted on the training data
    # not the validation data.
    model = sm.GLM(
        labelled["y_train"],
        labelled["x_train"],
        family=sm.families.Gamma(link=sm.families.links.Log())
    )
    results = model.fit()

    # This is the value of the confidence interval we ask
    # the model to report. Later we will check empirically
    # what proportion of our dataset actually falls into
    # this interval
    ci = 0.95

    # Use the model to predict the outputs for the inputs
    # it was trained on. If this is very inaccurate it
    # likely means that the model architecture (e.g.,
    # the family, link function, or choice of design
    # vectors) is poor.
    train_pred = results.get_prediction(
        labelled["x_train"]
    ).summary_frame(alpha=1-ci)

    # Use the model to predict the outputs for the inputs
    # it *hasn't* seen before. If this is significantly
    # worse than `train_pred`, this is a sign of overfitting.
    val_pred = results.get_prediction(
        labelled["x_val"]
    ).summary_frame(alpha=1-ci)

    # We can measure how accurate both of the above sets of
    # predictions are by counting the proportion of the results
    # fall within the reported confidence interval.
    train_proportion_in_ci = np.mean(
        (train_pred["mean_ci_lower"] >= labelled["y_train"]) &
        (labelled["y_train"] <= train_pred["mean_ci_upper"])
    )
    val_proportion_in_ci = np.mean(
        (val_pred["mean_ci_lower"] >= labelled["y_val"]) &
        (labelled["y_val"] <= val_pred["mean_ci_upper"])
    )

    if (train_proportion_in_ci - val_proportion_in_ci) > 0.1:
        print("Warning: Significant evidence of overfitting. Try reducing train_prop")

    # Now that we have a model and an indication of its
    # accuracy, we can use it to predict the given property.
    test_poi_box = assess.calculate_box(
        lat,
        lng,
        poi_radius_km,
        date,
        0   # `size_days` is zero because we want OSM data for
            # only the given day
    )

    test_pois = access.fetch_pois(test_poi_box, tags, verbose=verbose)
    test_poi_counts = assess.count_pois(test_pois, test_poi_box, tags)
    test_closest_city = assess.get_closest_city_to(lat, lng, df)
    test_distance_to_closest_city = assess.distance_km_to_city_center(
        lat,
        lng,
        test_closest_city
    )
    test_design = assess.make_design(
        [lat],
        [lng],
        [date],
        [property_type],
        np.array(list([test_poi_counts[date][tag]] for tag in tags)).T,
        [test_distance_to_closest_city]
    )

    test_pred = results.get_prediction(
        test_design
    ).summary_frame(alpha=1-ci)

    return {
        "estimate": test_pred["mean"][0],
        "upper_bound": test_pred["mean_ci_upper"][0],
        "lower_bound": test_pred["mean_ci_lower"][0],
        "confidence": val_proportion_in_ci,
    }
