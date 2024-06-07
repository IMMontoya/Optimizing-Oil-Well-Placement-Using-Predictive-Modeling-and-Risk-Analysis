# Optimizing Oil Well Placement Using Predictive Modeling and Risk Analysis

## Project Overview

This project aims to assist the OilyGiant mining company in identifying the optimal location for new oil well development of 200 wells. By analyzing geological data from three different regions, we build a predictive model to estimate the volume of oil reserves in potential new wells. The primary goal is to maximize profit by selecting the region with the highest expected returns while also accounting for risks. This is done by leveraging linear regression for predictions and employing bootstrapping techniques to determine the average predicted revenue and associated risks.

[Notebook](notebook.ipynb)  
[Data Description](#data-description)  
[Train and Test Model for Each Region](#train-and-test-the-model-for-each-region)  
[Prepare for Profit Calculation](#prepare-for-profit-calculation)  
[Calculate Profit](#calculate-profit)  
[Calculate Risks and Bootstrap Profit](#calculate-risks-and-bootstrap-profit)
[Final Recommendation](#final-recommendation)

## Data Description

The data for this project consists of geological exploration data from three different regions. Each dataset contains information about 100,000 potential oil wells. The data is synthetic, designed for the purpose of this project, and does not disclose actual contract details or well characteristics.

### Datasets

The project uses the following datasets:

- `geo_data_0.csv`
- `geo_data_1.csv`
- `geo_data_2.csv`

### Features

Each dataset includes the following columns:

- `id`: A unique identifier for each oil well.
- `f0`, `f1`, `f2`: Three features representing specific characteristics of the oil wells. The exact nature of these features is not disclosed, but they are significant for predicting oil reserves.
- `product`: The volume of oil reserves in the well (in thousand barrels).

### Data Preparation

The data preparation process involves the following steps:

1. **Loading the Data**: Read the datasets into a pandas DataFrame.
2. **Handling Missing Values**: There are no missing values, but there are some unexpected duplicate IDs. Because the features and product values for this instances differ between the duplicate instance, duplicates were treated as unique records.
3. **Feature Selection**: Select relevant features (`f0`, `f1`, `f2`) for modeling, along with the target variable (`product`).

### Summary Statistics

Here are some summary statistics for the features in each dataset:

#### geo_data_0.csv

|       | f0           | f1           | f2           |
|-------|--------------|--------------|--------------|
| count | 100000       | 100000       | 100000       |
| mean  | 0.500419     | 0.250143     | 2.502647     |
| std   | 0.871832     | 0.504433     | 3.248248     |
| min   | -1.408605    | -0.848218    | -12.088328   |
| 25%   | -0.072580    | -0.200881    | 0.287748     |
| 50%   | 0.502360     | 0.250252     | 2.515969     |
| 75%   | 1.073581     | 0.700646     | 4.715088     |
| max   | 2.362331     | 1.343769     | 16.003790    |

#### geo_data_1.csv

|       | f0           | f1           | f2           |
|-------|--------------|--------------|--------------|
| count | 100000       | 100000       | 100000       |
| mean  | 1.141296     | -4.796579    | 2.494541     |
| std   | 8.965932     | 5.119872     | 1.703572     |
| min   | -31.609576   | -26.358598   | -0.018144    |
| 25%   | -6.298551    | -8.267985    | 1.000021     |
| 50%   | 1.153055     | -4.813172    | 2.011479     |
| 75%   | 8.621015     | -1.332816    | 3.999904     |
| max   | 29.421755    | 18.734063    | 5.019721     |

#### geo_data_2.csv

|       | f0           | f1           | f2           |
|-------|--------------|--------------|--------------|
| count | 100000       | 100000       | 100000       |
| mean  | 0.002023     | -0.002081    | 2.495128     |
| std   | 1.732045     | 1.730417     | 3.473445     |
| min   | -8.760004    | -7.084020    | -11.970335   |
| 25%   | -1.162288    | -1.174820    | 0.130359     |
| 50%   | 0.009424     | -0.009482    | 2.484236     |
| 75%   | 1.158535     | 1.163678     | 4.858794     |
| max   | 7.238262     | 7.844801     | 16.739402    |

These statistics provide an overview of the distribution and range of the features, indicating the need for scaling due to differing scales and variances.

### Train and Test the Model for Each Region

In this section, we focus on building and evaluating a predictive model for each of the three regions using linear regression. The steps involved are as follows:

1. **Data Splitting**:
    - For each region, the data is split into training and validation sets in a 75:25 ratio. This ensures that the model is trained on a majority of the data but validated on a separate set to evaluate its performance on unseen data.

2. **Model Training**:
    - A linear regression model is trained on the training set for each region. Linear regression is chosen because it is simple and interpretable, and is specified by the project's requirements.

3. **Predictions & Model Evaluation**:
    - The trained model is then used to make predictions on the validation set. These predictions represent the estimated volume of oil reserves for the wells in the validation set.
    - The performance of the model is evaluated using the Root Mean Squared Error (RMSE).
    - The average predicted volume of reserves and the RMSE for each region are:  

            Region 0:
            Average volume of predicted reserves: 92.59256778438035
            RMSE: 37.5794217150813

            Region 1:
            Average volume of predicted reserves: 68.728546895446
            RMSE: 0.8930992867756163

            Region 2:
            Average volume of predicted reserves: 94.96504596800489
            RMSE: 40.02970873393434

4. **Conclusions**:  
    Regions 0 and 2 predicted the highest volume of predicted reserves, but also had very high RMSE values (prone to error) with values of 38 and 40 respectively. Region 1, predicted significantly lower reserves, but with a much better RMSE value of .89. This highlights the prescribed model's efficiency in predicting reserves at region 1 compared to regions 0 and 2.

### Prepare for Profit Calculation

In this section, we prepare the necessary calculations to estimate the profitability of developing 200 new oil wells in each region.

1. The minimum average volume of reserves necessary for development without losses is calculated as such:

    ```python
    # Calculate the volume of reserves sufficient for developing 200 new wells without losses.
    budget = 100_000_000 # budget for 200 wells development
    price_per_1k_barrel = 4_500 # revenue
    number_of_wells = 200

    min_reserve_volume = budget / price_per_1k_barrel / number_of_wells
    ```

    Minimum average volume of reserves for development without losses: 111.111111 (1K barrels)

2. This values is then compared to the average volume of reserves and the average volume of reserves for the top 200 wells in each region:

    Region 0 average volume of reserves: 92.50000000000001  
    Region 1 average volume of reserves: 68.82500000000002  
    Region 2 average volume of reserves: 95.00000000000004  

    Region 0 average volume of reserves for top 200 wells: 184.83373964536008  
    Region 1 average volume of reserves for top 200 wells: 137.9454077409057  
    Region 2 average volume of reserves for top 200 wells: 189.55147698176646

3. **Conclusions**:  
    The average volume of reserves necessary for profitability per oil well is 111.1111. The average volume for region 0 and 2 is slightly shy of that number compared to region 1, which misses the mark by about half. However, when we look at the average volume of reserves for the top 200 wells in each region, each region is well above the 111 threshold. Presumably, these regions are all profitable, which speaks to the inaccuracy of predicting profitability based off of averages in this way.

### Calculate Profit

1. Profits for each region are calculated by selecting the top 200 wells in terms of model prediction, then taking the actual reserves found in those wells and multiplying their sum by the $4,500.00 price per 1k barrels (1 unit of the product) with the budget amount subtracted. 

    ```python
    def profit(predictions):
    # Pick top 200 wells in terms of predicted reserves
    top_wells = predictions.sort_values('predictions', ascending=False).head(200)
    
    # Calculate profit
    profit = top_wells['actual'].sum() * price_per_1k_barrel - budget
    
    return profit
    ```

        Region 0 profit: 33,208,260.43  
        Region 1 profit: 24,150,866.97  
        Region 2 profit: 27,103,499.64  

2. **Conclusions**:  
Without consideration for risk, each region would expected to be profitable according to this model, with region 0 promising the most profitability and region 1 promising the least. However, it is important to keep in mind the high RMSE scores for region 0 and 2. The consequences of those high RMSE scores will become clear in the next section.

### Calculate Risks and Bootstrap Profit

1. Bootstrap methodology is used to make 1,000 profit predictions with a sample size of 500. Each prediction is stored in the values variable for calculating the 95% confidence interval.
    ```python
    def bootstrap(predictions):
    """
    Input: DataFrame with predictions and actual values for 'product'
    Output: List of 1000 profit values
    """
    values = []
    for i in range(1000):
        # Sample 500 wells with replacement
        sample = predictions.sample(n=500, replace=True, random_state=state)
        
        # Pick top 200 wells in terms of predicted reserves
        sample = sample.sort_values('predictions', ascending=False).head(200)
        
        # Calculate profit
        values.append(profit(sample))
        
    return values
        
    ```

2. The distribution of predicted profits for each region is plotted. The Average predicted profit, 95% confidence interval, and risk of losses are printed.  

        Region 0:  
        Average profit: $3,961,649.85  
        95% confidence interval: $-1,112,155.46 - $9,097,669.42  
        Risk of losses: 6.9%  
    ![Region 0 Profit Predictions Distribution](/images/Region0_prof_dist.png)

        Region 1:
        Average profit: $4,611,558.17
        95% confidence interval: $780,508.11 - $8,629,520.60
        Risk of losses: 0.7%
    ![Region 1 Profit Predictions Distribution](/images/Region1_prof_dist.png)

        Region 2:
        Average profit: $3,929,504.75
        95% confidence interval: $-1,122,276.25 - $9,345,629.15
        Risk of losses: 6.5%
    ![Region 2 Profit Predictions Distribution](/images/Region2_prof_dist.png)

### Final Recommendation

The best region for development is region 1. Not only does region 1 yield the highest profits based off the model's predictions, it does so with a higher degree of accuracy. This results in the least risk of losses (1.4%) among the three regions considered, and a 95% confidence interval that is entirely in the green (not the case for regions 0 and 2).  

In short, region 1 has the highest potential for returns, and lowest potential for losses among the three regions considered.
