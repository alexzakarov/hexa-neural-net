package services

import (
	"main/internal/neural_net/application/services/solver"
	"main/internal/neural_net/domain/ports"
	"main/internal/neural_net/domain/utils"
	"sync"
	"time"
)

// BatchTrainer implements parallelized batch training
type BatchTrainer struct {
	*internalb
	neuralNet   ports.INeuralNet
	verbosity   int
	batchSize   int
	parallelism int
	solver      solver.Solver
	printer     *StatsPrinter
}

type internalb struct {
	deltas            [][][]float64
	partialDeltas     [][][][]float64
	accumulatedDeltas [][][]float64
	moments           [][][]float64
}

func newBatchTraining(layers []ports.ILayer, parallelism int) *internalb {
	deltas := make([][][]float64, parallelism)
	partialDeltas := make([][][][]float64, parallelism)
	accumulatedDeltas := make([][][]float64, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]float64, len(layers))
		partialDeltas[w] = make([][][]float64, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]float64, len(l.GetNeurons()))
			accumulatedDeltas[i] = make([][]float64, len(l.GetNeurons()))
			partialDeltas[w][i] = make([][]float64, len(l.GetNeurons()))
			for j, n := range l.GetNeurons() {
				partialDeltas[w][i][j] = make([]float64, len(n.GetIn()))
				accumulatedDeltas[i][j] = make([]float64, len(n.GetIn()))
			}
		}
	}
	return &internalb{
		deltas:            deltas,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
}

// NewBatchTrainer returns a BatchTrainer
func NewBatchTrainer(n ports.INeuralNet, solver solver.Solver, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		neuralNet:   n,
		solver:      solver,
		verbosity:   verbosity,
		batchSize:   utils.Iparam(batchSize, 1),
		parallelism: utils.Iparam(parallelism, 1),
		printer:     NewStatsPrinter(),
	}
}

// Train trains n
func (t *BatchTrainer) Train(examples, validation Examples, iterations int) {
	t.internalb = newBatchTraining(t.neuralNet.GetLayers(), t.parallelism)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Example, t.parallelism)
	nets := make([]ports.INeuralNet, t.parallelism)

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = NewNeural(t.neuralNet.GetConfig())

		go func(id int, workCh <-chan Example) {
			n := nets[id]
			for e := range workCh {
				n.Forward(e.Input)
				t.calculateDeltas(e.Response, id)
				wg.Done()
			}
		}(i, workCh)
	}

	t.printer.Init(t.neuralNet)
	t.solver.Init(t.neuralNet.NumWeights())

	ts := time.Now()
	for it := 1; it <= iterations; it++ {
		train.Shuffle()
		batches := train.SplitSize(t.batchSize)

		for _, b := range batches {
			currentWeights := t.neuralNet.Weights()
			for _, n := range nets {
				n.ApplyWeights(currentWeights)
			}

			wg.Add(len(b))
			for _, item := range b {
				workCh <- item
			}
			wg.Wait()

			for _, wPD := range t.partialDeltas {
				for i, iPD := range wPD {
					iAD := t.accumulatedDeltas[i]
					for j, jPD := range iPD {
						jAD := iAD[j]
						for k, v := range jPD {
							jAD[k] += v
							jPD[k] = 0
						}
					}
				}
			}

			t.update(t.neuralNet, it)
		}

		if t.verbosity > 0 && it%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(t.neuralNet, validation, time.Since(ts), it)
		}
	}
}

func (t *BatchTrainer) calculateDeltas(ideal []float64, wid int) {
	loss := GetLoss(t.neuralNet.GetConfig().Loss)
	deltas := t.deltas[wid]
	partialDeltas := t.partialDeltas[wid]
	lastDeltas := deltas[len(t.neuralNet.GetLayers())-1]

	for i, n := range t.neuralNet.GetLayers()[len(t.neuralNet.GetLayers())-1].GetNeurons() {
		lastDeltas[i] = loss.Df(
			n.GetValue(),
			ideal[i],
			n.DActivate(n.GetValue()))
	}

	for i := len(t.neuralNet.GetLayers()) - 2; i >= 0; i-- {
		l := t.neuralNet.GetLayers()[i]
		iD := deltas[i]
		nextD := deltas[i+1]
		for j, n := range l.GetNeurons() {
			var sum float64
			for k, s := range n.GetOut() {
				sum += s.GetWeight() * nextD[k]
			}
			iD[j] = n.DActivate(n.GetValue()) * sum
		}
	}

	for i, l := range t.neuralNet.GetLayers() {
		iD := deltas[i]
		iPD := partialDeltas[i]
		for j, n := range l.GetNeurons() {
			jD := iD[j]
			jPD := iPD[j]
			for k, s := range n.GetIn() {
				jPD[k] += jD * s.GetIn()
			}
		}
	}
}

func (t *BatchTrainer) update(n ports.INeuralNet, it int) {
	var idx int
	for i, l := range n.GetLayers() {
		iAD := t.accumulatedDeltas[i]
		for j, n := range l.GetNeurons() {
			jAD := iAD[j]
			for k, s := range n.GetIn() {
				update := t.solver.Update(s.GetWeight(),
					jAD[k],
					it,
					idx)
				s.SetWeight(s.GetWeight() + update)
				jAD[k] = 0
				idx++
			}
		}
	}
}
