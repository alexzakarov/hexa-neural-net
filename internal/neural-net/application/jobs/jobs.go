package jobs

import (
	"context"
	"fmt"
	"main/config"
	"main/internal/neural-net/domain/ports"
	"main/pkg/logger"
)

// jobRunner Indicator Worker struct
type jobRunner struct {
	cfg    *config.Config
	logger logger.Logger
	srv    ports.IService
}

// NewJobRunner Indicator worker constructor
func NewJobRunner(cfg *config.Config, logger logger.Logger, srv ports.IService) ports.IJobs {
	return &jobRunner{
		cfg:    cfg,
		logger: logger,
		srv:    srv,
	}
}

func (w *jobRunner) TrainNeuralNet(ctx context.Context) {
	fmt.Println("test")
}
