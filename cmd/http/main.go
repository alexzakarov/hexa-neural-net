package main

import (
	"context"
	"log"
	"main/config"
	_ "main/docs"
	neuralNetJobs "main/internal/neural-net/application/jobs"
	neuralNetServices "main/internal/neural-net/application/services"
	neuralNetHandlers "main/internal/neural-net/handler/http"
	neuralNetRepos "main/internal/neural-net/infrastructure/repository"
	"main/pkg/databases/postgresql"
	"main/pkg/logger"
	"main/pkg/server"
	"main/pkg/utils/graceful_exit"
)

// @title Auth Service
// @version 1.0
// @description Common Auth service broker with REST endpoints
// @contact.email ivanbarayev@hotmail.com
// @BasePath /v1
func main() {
	log.Println("Starting api server")

	cfg, errConfig := config.ParseConfig()
	if errConfig != nil {
		log.Fatal(errConfig)
	}

	appLogger := logger.NewApiLogger(cfg)

	appLogger.InitLogger()
	appLogger.Infof("AppVersion: %s, LogLevel: %s, Mode: %s", cfg.Server.APP_VERSION, cfg.Logger.LEVEL, cfg.Server.APP_ENV)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Init Clients
	postgresqlDB, err := postgresql.NewPostgresqlDB(cfg)
	if err != nil {
		appLogger.Fatal("Error when tyring to connect to Postgresql")
	} else {
		appLogger.Info("Postgresql connected")
	}

	// Init repositories
	pgRepo := neuralNetRepos.NewPostgresqlRepository(postgresqlDB)

	// Init services
	neuralNetService := neuralNetServices.NewNeuralNetService(cfg, pgRepo, appLogger)

	//Init jobs
	neuraLNetJobs := neuralNetJobs.NewJobRunner(cfg, appLogger, neuralNetService)

	// Interceptors
	//

	servers := server.NewServer(cfg, &ctx, appLogger)

	httpServer, errHttpServer := servers.NewHttpServer()
	if errHttpServer != nil {
		println(errHttpServer.Error())
	}
	versioning := httpServer.Group("/v1")

	// Init handlers for HTTP Server
	neuralNetHandler := neuralNetHandlers.NewHttpHandler(ctx, cfg, neuralNetService, appLogger)

	// Init routes for HTTP Server
	neuralNetHandlers.MapRoutes(neuralNetHandler, versioning)

	//telegram.SendMessage("Send Message to telegram channel")

	//Start Jobs
	go neuraLNetJobs.TrainNeuralNet(ctx)

	// Exit from application gracefully
	graceful_exit.TerminateApp(ctx)

	appLogger.Info("Server Exited Properly")
}
