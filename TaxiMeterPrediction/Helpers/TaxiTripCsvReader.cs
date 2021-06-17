using Regression_TaxiFarePrediction.DataStructures;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace TaxiMeterPrediction.Helpers
{
    public class TaxiTripCsvReader
    {
        public IEnumerable<TaxiTrip> GetDataFromCsv(string dataLocation, int numMaxRecords)
        {
            IEnumerable<TaxiTrip> records =
                File.ReadAllLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new TaxiTrip()
                {
                    VendorId = x[0],
                    RateCode = x[1],
                    PassengerCount = float.Parse(x[2], CultureInfo.InvariantCulture),
                    TripTime = float.Parse(x[3], CultureInfo.InvariantCulture),
                    TripDistance = float.Parse(x[4], CultureInfo.InvariantCulture),
                    PaymentType = x[5],
                    FareAmount = float.Parse(x[6], CultureInfo.InvariantCulture)
                })
                .Take<TaxiTrip>(numMaxRecords);

            return records;
        }
    }
}
