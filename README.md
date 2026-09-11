# 基于大模型的智能课程答疑系统

一个基于 DeepSeek 大模型和 Streamlit 构建的智能课程答疑系统，提供流畅的多会话问答体验。

## ✨ 核心特性

- **多会话隔离管理**：每个问答会话独立存储，互不干扰
- **实时流式答疑**：基于 SSE 技术，逐字渲染 AI 回答
- **历史自动保存**：所有对话自动持久化，随时查看历史会话
- **现代化界面**：学术靛蓝色系主题，简洁专注的交互设计

## 🚀 快速开始

### 前置要求

- Python 3.8+
- DeepSeek API Key ([获取地址](https://platform.deepseek.com/))

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/tec-zhh9/course-qa-system.git
   cd course-qa-system
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置 API Key**
   ```bash
   # 复制环境变量模板
   cp .env.example .env
   
   # 编辑 .env 文件，填入你的 DeepSeek API Key
   # DEEPSEEK_API_KEY=sk-your-api-key-here
   ```

4. **启动应用**
   ```bash
   streamlit run app.py
   ```

应用将在浏览器中自动打开 `http://localhost:8501`

## 📦 项目结构

```
course-qa-system/
├── app.py                 # 主应用程序
├── llm_client.py          # DeepSeek API 客户端封装
├── requirements.txt       # Python 依赖清单
├── .env.example           # 环境变量模板
├── .gitignore             # Git 忽略规则
├── .streamlit/
│   └── config.toml        # Streamlit 主题配置
└── sessions_store.json    # 会话历史数据（自动生成）
```

## 🛠️ 技术架构

- **前端框架**：Streamlit 1.63.0
- **AI 模型**：DeepSeek Chat API
- **HTTP 客户端**：requests + SSE 流式解析
- **数据存储**：JSON 文件（会话持久化）
- **配置管理**：python-dotenv

## ⚙️ 配置说明

### 环境变量

在 `.env` 文件中配置以下参数：

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API 密钥 |

### 主题定制

修改 `.streamlit/config.toml` 以自定义界面主题：

```toml
[theme]
primaryColor = "#4F46E5"      # 主题色
backgroundColor = "#FFFFFF"    # 背景色
secondaryBackgroundColor = "#F3F4F6"  # 次级背景色
textColor = "#1F2937"          # 文字颜色
font = "sans serif"            # 字体
```

## 🔒 安全提示

⚠️ **请勿将 `.env` 文件提交到版本控制系统**  
- 该文件包含敏感的 API 密钥
- 项目已通过 `.gitignore` 排除此文件
- 使用 `.env.example` 作为配置模板

## 📝 使用说明

1. **创建新会话**：点击主页"➕ 新建会话"按钮
2. **输入问题**：在输入框中输入课程相关问题
3. **查看回答**：AI 将实时流式输出答案
4. **历史查看**：侧边栏可切换和查看历史会话

## 🤝 致谢

- [DeepSeek](https://www.deepseek.com/) - 提供强大的大模型 API
- [Streamlit](https://streamlit.io/) - 提供优雅的 Web 应用框架
- [Trae CN](https://github.com/tec-zhh9) - 项目开发支持

## 📄 许可证

本项目仅供学习交流使用。
