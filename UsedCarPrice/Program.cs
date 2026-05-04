//*******************************************************************
// Philip Margallo
// CSE-41413
// Section 197521
// Husam El-Issa
// Week 5 Assignment
// Due Date: 5/3/2026
//*******************************************************************

using Microsoft.ML;
using UsedCarPrice;


// Create Microsoft ML context
var ml = new MLContext(seed: 1);
//var ml = new MLContext();

// The path of the used car data CSV file is relative
// to project
string path = "used_cars_1000.csv";

// 1. Load the data
IDataView data = ml.Data.LoadFromTextFile<UsedCarData>(
    path: path,
    hasHeader: true,
    separatorChar: ',');

//// Select preview the data
//var preview = data.Preview(maxRows: 5);

//// Display preview data
//foreach (var row in preview.RowView)
//{
//    foreach (var column in row.Values)
//    {
//        Console.Write($"{column.Key}: {column.Value} | ");
//    }
//    Console.WriteLine();
//}

// 2. Split into train/test sets
var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

// 3. Build the pipeline
//var pipeline = 
//    ml.Transforms.Categorical.OneHotEncoding(
//        new[]
//        {
//            new InputOutputColumnPair("MakeEncoded", "Make"),
//            new InputOutputColumnPair("ModelEncoded", "Model"),
//            new InputOutputColumnPair("TransmissionEncoded", "Transmission"),
//            new InputOutputColumnPair("FuelTypeEncoded", "FuelType")
//        })
//    .Append(ml.Transforms.Concatenate("Features",
//                                        "MakeEncoded",
//                                        "ModelEncoded", 
//                                        "Year", 
//                                        "Mileage", 
//                                        "EngineSize", 
//                                        "Horsepower",
//                                        "TransmissionEncoded",
//                                        "FuelTypeEncoded", 
//                                        "Doors"))
//    .Append(ml.Transforms.NormalizeMinMax("Features"))
//    .Append(ml.Regression.Trainers.Sdca(
//        labelColumnName: "Price",
//        featureColumnName: "Features"));

// Modified original Estimator implementation to use nameof() method
var pipeline =
    // Fit() was giving an error when processing the columns of type string,
    // so I used the OneHotEncoding to convert them to numeric values, as
    // suggested by Microsoft Copilot.
    ml.Transforms.Categorical.OneHotEncoding(
        new[]
        {
            new InputOutputColumnPair("MakeEncoded", nameof(UsedCarData.Make)),
            new InputOutputColumnPair("ModelEncoded", nameof(UsedCarData.Model)),
            new InputOutputColumnPair("TransmissionEncoded", nameof(UsedCarData.Transmission)),
            new InputOutputColumnPair("FuelTypeEncoded", nameof(UsedCarData.FuelType))
        })
    .Append(ml.Transforms.Concatenate("Features",
                                        "MakeEncoded",
                                        "ModelEncoded",
                                        nameof(UsedCarData.Year),
                                        nameof(UsedCarData.Mileage),
                                        nameof(UsedCarData.EngineSize),
                                        nameof(UsedCarData.Horsepower),
                                        "TransmissionEncoded",
                                        "FuelTypeEncoded",
                                        nameof(UsedCarData.Doors)))
    .Append(ml.Transforms.NormalizeMinMax("Features"))
    .Append(ml.Regression.Trainers.Sdca(
        labelColumnName: nameof(UsedCarData.Price),
        featureColumnName: "Features"));

// 4. Train the model on the training set
var trainedModel = pipeline.Fit(split.TrainSet);

// 5. Evaluate on the test set
var predictions = trainedModel.Transform(split.TestSet);

// Obtain metrics
var metrics = ml.Regression.Evaluate(
    predictions,
    labelColumnName: "Price");

// Display metrics
Console.WriteLine($"R²:  {metrics.RSquared:0.###}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.##}");
Console.WriteLine($"RMSE:{metrics.RootMeanSquaredError:0.##}");

// 6. Create a prediction engine for single predictions
var engine = ml.Model.CreatePredictionEngine<UsedCarData, UsedCarPrediction>(trainedModel);

// 7. Make a sample prediction
var sample = new UsedCarData
{
    Make = "Honda",
    Model = "Accord",
    Year = 1996,
    Mileage = 203500,
    EngineSize = 1.3F,
    Horsepower = 136,
    Transmission = "Automatic",
    FuelType = "Gasoline",
    Doors = 4
};

// Test data from the CSV file, which should yield a price close to 5055
// Honda, Accord,2013,187984,1.8,161, Automatic, Gasoline,4,5055

var prediction = engine.Predict(sample);

Console.WriteLine($"Predicted price: {prediction.Score:0}");