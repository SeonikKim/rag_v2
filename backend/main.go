package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.POST("/upload", uploadHandler)
	r.POST("/query", queryHandler)

	r.Run(":8080")
}

func uploadHandler(c *gin.Context) {
	// TODO: call Python pipeline to process PDF and store data
	c.JSON(http.StatusOK, gin.H{"status": "uploaded"})
}

func queryHandler(c *gin.Context) {
	// TODO: embed question, query Milvus and MariaDB, return result
	c.JSON(http.StatusOK, gin.H{"answer": ""})
}
