# Repository Overview

This repository contains my implementations for the tasks set out in the ADS assessment.

My approach to the problem stuck closely to the suggested structure:

## 1. Select a bounding box around the housing location in latitude and longitude.

To aid interpretability, I wanted to specify this bounding box using kilometres rather than e.g., degrees. The helper function (`calculate_box`) for doing this conversion are in `assess`. I ended up using a range of 10km on either side of the housing location (a 20km $\times$ 20km square). This is specified as a default parameter of the `predict_price` function (in `address`).

## 2. Select a date range around the prediction date.

I used a date range of 3 days before/after the prediction date (a range of 7 days total). This is specified as a default parameter of the `predict_price` function (in `address`). Together, the bounding box and the date range make up a `Box`, a data structure defined in `access`. Ideally this would have been defined in `assess` instead, but due to the methodology put forth in the description of Task D:

> To build our prediction model, we're going to use the prices for a particular region in a given time period. This means we can select that region and time period and build the joined data only from the relevent rows from the two tables.

A `Box` is required in order to do the actual fetching of data from the database.

## 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.

The `access` module contains generic reusable wrappers around PyMySQL to perform tasks such as connecting to a database (`make_connection`), executing SQL commands (`execute`), fetching the results from SQL queries (`query`), and uploading local CSV files into a table (`upload_file`).

It also contains a helper function (`upload_pp_data`) which downloads property and price data from ProPublica part-by-part and then uploads the data into the `pp_data` table.

Furthermore, the module contains a wrapper function (`join_pp_and_postcode_data`) to perform the inner join specified in part `D`.

On the OSM side of things, `access` contains a function (`fetch_pois`) which queries the OSM API for POIs with a set of tags within a given bounding box. Furthermore, such a query is performed for every date within a date range, as OSM allows you to query historical data for a given date.

Given all of this data, `assess` contains methods to play around with it. For example, given the aforementioned POI dataframe for each date in a date range, `assess` contains a function (`count_pois`) which counts how many there are of each type of POI within a given region on each date.

It also contains functions for operations such as finding the closest city to a given property (`get_closest_city_to`), and approximating the distance to the centre of that city (`distance_km_to_city_center`).

Finally, the functions for assembling all of this data into a form in which it can be used to fit a GLM are also in `assess`. The data is split into training data and validation data (by default, 80% training, 20% validation).

In the end, the design matrix I've used contains the following vectors:

 - The latitude itself
 - The longitude itself
 - The number of days between the prediction date and the start of the dataset (Jan 1st 1996)
 - A set of 5 one hot vectors indicating the property type
 - The number of `cuisine` POIs within 0.5km of the property on the prediction date
 - The number of `shop` POIs within 0.5km of the property on the prediction date
 - The number of `tourism` POIs within 0.5km of the property on the prediction date
 - The number of `healthcare` POIs within 0.5km of the property on the prediction date
 - The approximate distance from the property to the centre of the nearest city
 - A constant

## 4. Train a linear model on the data set you have created.

This step is in `predict_price` in `address`. The family of GLM I've used is Gamma. This is because I expect the nature of property prices to be such that most are small, and there are relatively few which are large. This is to say, I expect there to be a heavy skew in the data.

The link function I've used is `Log`, because I expect the magnitude of the price to hold a lot more significance than the absolute value.

The model is fitted on the training data obtained from `assess.labelled` (not the validation data).

## 5. Validate the quality of the model.

When the model is used to make predictions, it reports a confidence interval for each data point. I can then count the proportion of the ground truth values which fall inside the reported confidence interval.

## 6. Provide a prediction of the price from the model, warning appropriately if your validation indicates the quality of the model is poor.

I calculate the proportion described above for the training data and validation data separately. If the former is significantly greater (I used 10 percentage points) than the latter, this means that the model was much more accurate on the training data than the validation data, which is a sign of overfitting. This results in a warning being thrown.

The proportion for the validation data is returned along with the prediction as a measure of confidence in the prediction (i.e., the best guess at the probability that the true price falls within the predicted interval)

Furthermore, a warning is thrown if the bounding box and date range do not contain enough data points, as this will likely result in a poor quality model.
