const express = require("express");
const bodyParser = require("body-parser");

const app = express();
const PORT = 3000;

// JSON 형식의 요청 본문을 파싱하기 위한 미들웨어
app.use(bodyParser.json());

// 기본 경로에 대한 응답 추가
app.get("/", (req, res) => {
  res.send("Welcome to the Face Recognition Server");
});

// 얼굴 인식 데이터를 받는 라우트
app.post("/data", (req, res) => {
  const { user_name, confidence } = req.body;

  console.log(`Received data - User: ${user_name}, Confidence: ${confidence}`);
  res.status(200).send("Data received successfully");
});

// 서버 실행
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
