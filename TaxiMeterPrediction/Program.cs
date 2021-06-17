using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;

using PLplot;

using Regression_TaxiFarePrediction.DataStructures;
using TaxiMeterPrediction.Helpers;
using static Microsoft.ML.Transforms.NormalizingEstimator;

namespace TaxiMeterPrediction
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-train.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-test.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/TaxiFareModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            // Crear, Entrenar, Evaluar y Guardar el modelo
            BuildTrainEvaluateAndSaveModel(mlContext);

            // Predecir la tarifa de una viaje cargando el modelo desde el archivo .ZIP 
            TestSinglePrediction(mlContext);

            // Mostrar el chart de la distribución de los datos de la regresión para 100 datos leidos desde el csv
            PlotRegressionChart(mlContext, TestDataPath, 100, args);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();

        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // PASO 1: Carga de datos
            IDataView baseTrainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

            // PASO 2: Preparar los datos
            // Remover "outliers", usando la columna FareAmounts, filtrar los valores mayores a 150 y menores que $1. 
            // Seguramente es data erronea
            var cnt = baseTrainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();
            Console.WriteLine($"Nro registros antes de eliminar outliers: {cnt}");

            IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(TaxiTrip.FareAmount), lowerBound: 1, upperBound: 150);

            var cnt2 = trainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();
            Console.WriteLine($"Nro registros despues de eliminar outliers: {cnt2}");

            // Transformar columnas categóricas y normalizar los datos numéricos

            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TaxiTrip.FareAmount))

                // "One hot encoding" para transformar columnas categóricas
                //Rate Code(1,2,3)     VendorId (CMT,VTS)   
                // 1 0 0                   1 0
                // 1 0 0                   0 1
                // 0 1 0                   1 0
                // 0 1 0                   0 1
                // 0 0 1                   1 0
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiTrip.VendorId)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiTrip.RateCode)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiTrip.PaymentType)))

                // Normalizar el valor de las columnas   (X - promedio)/dev.std.
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.PassengerCount)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripTime)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripDistance)))

                // Finalmente devolvemos todas las columnas procesadas
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiTrip.PassengerCount), nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance)));

            // (Opcional) Revisar el resultado de la transformación de los datos (5 records)  
            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);
            ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 5);


            // PASO 3: Seleccionar el algoritmo de ML - SDCA Regression (Stochastic Dual Coordinate Ascent)
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // PASO 4: Entrenar el modelo utilizando "training data"
            // Este modelo es entrenado utilizando los datos cargados y tranformados
            Console.WriteLine("=============== Entrenando el modelo ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);


            // PASO 5: Evaluar el modelo entrenado, mostrando el "accuracy"
            Console.WriteLine("===== Evaluando 'accuracy' del Modelo con 'test data' =====");
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);


            // STEP 6: Guardar el modelo entrenado a un archivo .ZIP 
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            Console.WriteLine("El modelo ha sido guardado en {0}", ModelPath);

            return trainedModel;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            //Ejemplo: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD, -> 15.5 <-

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0  // la predicción irá aquí. Actual/Observado = 15.5
            };

            ///
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Crear 'prediction engine' desde el modelo cargado
            var predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

            //Predecir
            var resultprediction = predEngine.Predict(taxiTripSample);
            ///

            Console.WriteLine($"************************************************************************");
            Console.WriteLine($"Predicción : {resultprediction.FareAmount:0.####}, valor real: 15.5");
            Console.WriteLine($"************************************************************************");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }



        private static void PlotRegressionChart(MLContext mlContext,
                                                        string testDataSetPath,
                                                        int numberOfRecordsToRead,
                                                        string[] args)
        {
            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            // Create prediction engine related to the loaded trained model
            var predFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

            string chartFileName = "";
            using (var pl = new PLStream())
            {
                // use SVG backend and write to SineWaves.svg in current directory
                if (args.Length == 1 && args[0] == "svg")
                {
                    pl.sdev("svg");
                    chartFileName = "TaxiRegressionDistribution.svg";
                    pl.sfnam(chartFileName);
                }
                else
                {
                    pl.sdev("pngcairo");
                    chartFileName = "TaxiRegressionDistribution.png";
                    pl.sfnam(chartFileName);
                }

                // use white background with black foreground
                pl.spal0("cmap0_alternate.pal");

                // Initialize plplot
                pl.init();

                // set axis limits
                const int xMinLimit = 0;
                const int xMaxLimit = 35; //Rides larger than $35 are not shown in the chart
                const int yMinLimit = 0;
                const int yMaxLimit = 35;  //Rides larger than $35 are not shown in the chart
                pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

                // Set scaling for mail title text 125% size of default
                pl.schr(0, 1.25);

                // The main title
                pl.lab("Measured", "Predicted", "Distribution of Taxi Fare Prediction");

                // plot using different colors
                // see http://plplot.sourceforge.net/examples.php?demo=02 for palette indices
                pl.col0(1);

                int totalNumber = numberOfRecordsToRead;
                var testData = new TaxiTripCsvReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();

                //This code is the symbol to paint
                char code = (char)9;

                // plot using other color
                //pl.col0(9); //Light Green
                //pl.col0(4); //Red
                pl.col0(2); //Blue

                double yTotal = 0;
                double xTotal = 0;
                double xyMultiTotal = 0;
                double xSquareTotal = 0;

                for (int i = 0; i < testData.Count; i++)
                {
                    var x = new double[1];
                    var y = new double[1];

                    //Make Prediction
                    var FarePrediction = predFunction.Predict(testData[i]);

                    x[0] = testData[i].FareAmount;
                    y[0] = FarePrediction.FareAmount;

                    //Paint a dot
                    pl.poin(x, y, code);

                    xTotal += x[0];
                    yTotal += y[0];

                    double multi = x[0] * y[0];
                    xyMultiTotal += multi;

                    double xSquare = x[0] * x[0];
                    xSquareTotal += xSquare;

                    double ySquare = y[0] * y[0];

                    Console.WriteLine($"-------------------------------------------------");
                    Console.WriteLine($"Predicted : {FarePrediction.FareAmount}");
                    Console.WriteLine($"Actual:    {testData[i].FareAmount}");
                    Console.WriteLine($"-------------------------------------------------");
                }

                // Regression Line calculation explanation:
                // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

                double minY = yTotal / totalNumber;
                double minX = xTotal / totalNumber;
                double minXY = xyMultiTotal / totalNumber;
                double minXsquare = xSquareTotal / totalNumber;

                double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);

                double b = minY - (m * minX);

                //Generic function for Y for the regression line
                // y = (m * x) + b;

                double x1 = 1;
                //Function for Y1 in the line
                double y1 = (m * x1) + b;

                double x2 = 39;
                //Function for Y2 in the line
                double y2 = (m * x2) + b;

                var xArray = new double[2];
                var yArray = new double[2];
                xArray[0] = x1;
                yArray[0] = y1;
                xArray[1] = x2;
                yArray[1] = y2;

                pl.col0(4);
                pl.line(xArray, yArray);

                // end page (writes output to disk)
                pl.eop();

                // output version of PLplot
                pl.gver(out var verText);
                Console.WriteLine("PLplot version " + verText);

            } // the pl object is disposed here

            // Open Chart File In Microsoft Photos App (Or default app, like browser for .svg)

            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }

    }
}
