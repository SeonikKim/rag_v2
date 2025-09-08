// 간단한 RAG 백엔드 서버
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
        // TODO: Python 파이프라인을 호출하여 PDF를 처리하고 데이터를 저장
        c.JSON(http.StatusOK, gin.H{"status": "uploaded"})
}

func queryHandler(c *gin.Context) {
        // TODO: 질문을 임베딩하고 Milvus와 MariaDB를 조회하여 결과를 반환
        c.JSON(http.StatusOK, gin.H{"answer": ""})
}
