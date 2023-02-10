package services

import (
	"main/config"
	"main/internal/neural-net/domain/ports"
	"main/pkg/logger"
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
