package services

import (
	"encoding/json"
	"fmt"
	"main/config"
	"main/internal/neural_net/application/services/layer/neuron/synapse"
	"main/internal/neural_net/application/services/solver"
	"main/internal/neural_net/domain/entities"
	"main/internal/neural_net/domain/ports"
	"main/pkg/logger"
	"main/pkg/utils/typeconv"
	"math/rand"
)

var (
	err error
)

const (
	collection = "examples"
)

// serviceNeuralNet Auth Service
type serviceNeuralNet struct {
	cfg    *config.Config
	pgRepo ports.IPostgresqlRepository
	logger logger.Logger
}

// NewNeuralNetService Auth domain service constructor
func NewNeuralNetService(cfg *config.Config, pgRepo ports.IPostgresqlRepository, logger logger.Logger) ports.IService {
	return &serviceNeuralNet{cfg: cfg, pgRepo: pgRepo, logger: logger}
}

func (w *serviceNeuralNet) Train() {
	rand.Seed(0)
	n := NewNeural(&entities.Config{
		Inputs:     4,
		Layout:     []int{5, 1}, // Sufficient for modeling (AND+OR) - with 5-6 neuron always converges
		Activation: entities.ActivationSigmoid,
		Mode:       entities.ModeBinary,
		Weight:     synapse.NewUniform(1, 0),
		Bias:       true,
	})
	permutations := GetData()

	trainer := NewTrainer(solver.NewSGD(0.1, 0.1, 1e-6, false), 50)
	trainer.Train(n, permutations, permutations, 10000)
	fmt.Println(n.Predict(permutations[0].Input))
}

func GetData() (Linedata Examples) {
	data := `[
  {
    "openTime": 1502928000000,
    "open": "4261.48000000",
    "high": "4485.39000000",
    "low": "4261.32000000",
    "close": "4427.30000000",
    "volume": "145.70874700",
    "closeTime": 1502971199999,
    "quoteAssetVolume": "635695.49067993",
    "trades": 582,
    "takerBaseAssetVolume": "122.80136000",
    "takerQuoteAssetVolume": "536701.47306474",
    "ignored": "0"
  },
  {
    "openTime": 1502971200000,
    "open": "4436.06000000",
    "high": "4485.39000000",
    "low": "4200.74000000",
    "close": "4285.08000000",
    "volume": "649.44163000",
    "closeTime": 1503014399999,
    "quoteAssetVolume": "2819074.56005213",
    "trades": 2845,
    "takerBaseAssetVolume": "493.44718100",
    "takerQuoteAssetVolume": "2141514.92753927",
    "ignored": "0"
  },
  {
    "openTime": 1503014400000,
    "open": "4285.08000000",
    "high": "4371.52000000",
    "low": "4134.61000000",
    "close": "4340.31000000",
    "volume": "720.72220100",
    "closeTime": 1503057599999,
    "quoteAssetVolume": "3085356.16659987",
    "trades": 3051,
    "takerBaseAssetVolume": "585.40800100",
    "takerQuoteAssetVolume": "2508148.14656605",
    "ignored": "0"
  },
  {
    "openTime": 1503057600000,
    "open": "4320.52000000",
    "high": "4340.31000000",
    "low": "3938.77000000",
    "close": "4108.37000000",
    "volume": "479.16606300",
    "closeTime": 1503100799999,
    "quoteAssetVolume": "2001602.13957164",
    "trades": 2182,
    "takerBaseAssetVolume": "387.46070900",
    "takerQuoteAssetVolume": "1620975.16995203",
    "ignored": "0"
  },
  {
    "openTime": 1503100800000,
    "open": "4108.37000000",
    "high": "4184.69000000",
    "low": "3850.00000000",
    "close": "3957.60000000",
    "volume": "298.51856900",
    "closeTime": 1503143999999,
    "quoteAssetVolume": "1214382.54455495",
    "trades": 1559,
    "takerBaseAssetVolume": "213.29778500",
    "takerQuoteAssetVolume": "870668.01762263",
    "ignored": "0"
  },
  {
    "openTime": 1503144000000,
    "open": "3945.12000000",
    "high": "4149.99000000",
    "low": "3928.89000000",
    "close": "4139.98000000",
    "volume": "82.79119400",
    "closeTime": 1503187199999,
    "quoteAssetVolume": "335101.19086656",
    "trades": 594,
    "takerBaseAssetVolume": "61.03825700",
    "takerQuoteAssetVolume": "247333.85246472",
    "ignored": "0"
  },
  {
    "openTime": 1503187200000,
    "open": "4120.98000000",
    "high": "4211.08000000",
    "low": "4032.62000000",
    "close": "4106.53000000",
    "volume": "71.23513600",
    "closeTime": 1503230399999,
    "quoteAssetVolume": "293071.51187086",
    "trades": 622,
    "takerBaseAssetVolume": "46.16552700",
    "takerQuoteAssetVolume": "189987.72648683",
    "ignored": "0"
  },
  {
    "openTime": 1503230400000,
    "open": "4106.53000000",
    "high": "4185.94000000",
    "low": "4063.35000000",
    "close": "4086.29000000",
    "volume": "395.84788600",
    "closeTime": 1503273599999,
    "quoteAssetVolume": "1637292.87845560",
    "trades": 1699,
    "takerBaseAssetVolume": "330.63042000",
    "takerQuoteAssetVolume": "1367413.60725047",
    "ignored": "0"
  },
  {
    "openTime": 1503273600000,
    "open": "4069.13000000",
    "high": "4119.62000000",
    "low": "3953.40000000",
    "close": "4023.11000000",
    "volume": "522.29158600",
    "closeTime": 1503316799999,
    "quoteAssetVolume": "2117739.09106764",
    "trades": 2585,
    "takerBaseAssetVolume": "435.16468900",
    "takerQuoteAssetVolume": "1764902.81501784",
    "ignored": "0"
  },
  {
    "openTime": 1503316800000,
    "open": "4023.11000000",
    "high": "4070.49000000",
    "low": "3911.79000000",
    "close": "4016.00000000",
    "volume": "169.45147400",
    "closeTime": 1503359999999,
    "quoteAssetVolume": "679492.62295964",
    "trades": 1387,
    "takerBaseAssetVolume": "122.19141800",
    "takerQuoteAssetVolume": "490759.73814053",
    "ignored": "0"
  },
  {
    "openTime": 1503360000000,
    "open": "4016.00000000",
    "high": "4016.00000000",
    "low": "3400.00000000",
    "close": "3866.48000000",
    "volume": "503.63810100",
    "closeTime": 1503403199999,
    "quoteAssetVolume": "1924565.61357909",
    "trades": 3132,
    "takerBaseAssetVolume": "327.14391100",
    "takerQuoteAssetVolume": "1253199.53186362",
    "ignored": "0"
  },
  {
    "openTime": 1503403200000,
    "open": "3866.48000000",
    "high": "4104.82000000",
    "low": "3786.81000000",
    "close": "4040.00000000",
    "volume": "463.04675700",
    "closeTime": 1503446399999,
    "quoteAssetVolume": "1827940.15856142",
    "trades": 3362,
    "takerBaseAssetVolume": "96.85127000",
    "takerQuoteAssetVolume": "383988.83747864",
    "ignored": "0"
  },
  {
    "openTime": 1503446400000,
    "open": "4040.00000000",
    "high": "4242.32000000",
    "low": "4013.89000000",
    "close": "4237.99000000",
    "volume": "436.52200900",
    "closeTime": 1503489599999,
    "quoteAssetVolume": "1791941.26220467",
    "trades": 3161,
    "takerBaseAssetVolume": "86.89992200",
    "takerQuoteAssetVolume": "359251.87383735",
    "ignored": "0"
  },
  {
    "openTime": 1503489600000,
    "open": "4205.98000000",
    "high": "4265.80000000",
    "low": "4069.80000000",
    "close": "4114.01000000",
    "volume": "564.61455600",
    "closeTime": 1503532799999,
    "quoteAssetVolume": "2356745.20361501",
    "trades": 5468,
    "takerBaseAssetVolume": "222.51917000",
    "takerQuoteAssetVolume": "934315.22135728",
    "ignored": "0"
  },
  {
    "openTime": 1503532800000,
    "open": "4147.00000000",
    "high": "4250.94000000",
    "low": "4085.01000000",
    "close": "4202.00000000",
    "volume": "418.18609800",
    "closeTime": 1503575999999,
    "quoteAssetVolume": "1739475.02544040",
    "trades": 3406,
    "takerBaseAssetVolume": "121.77274200",
    "takerQuoteAssetVolume": "508853.09145766",
    "ignored": "0"
  }]`
	var arr []map[string]interface{}
	err := json.Unmarshal([]byte(data), &arr)

	if err != nil {
		println(err.Error())
	}
	Linedata = ChartDataParser(arr)
	return
}

func ChartDataParser(arr []map[string]interface{}) (Linedata Examples) {
	fmt.Println()

	for i := 0; i < len(arr); i++ {
		data := Example{
			[]float64{
				typeconv.ToFloat(arr[i]["open"]),
				typeconv.ToFloat(arr[i]["high"]),
				typeconv.ToFloat(arr[i]["low"]),
				typeconv.ToFloat(arr[i]["close"]),
			}, []float64{
				arr[i]["closeTime"].(float64),
			},
		}
		Linedata = append(Linedata, data)
	}
	return
}
