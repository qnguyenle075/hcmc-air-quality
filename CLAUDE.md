# CLAUDE.md — HCMC Air Quality Agentic RAG

> File này vừa là **rule** cho Claude Code, vừa là **plan triển khai** của project.
> Claude Code: đọc toàn bộ file này trước khi làm bất kỳ task nào. Luôn làm theo đúng thứ tự phase, không nhảy cóc.

---

## 0. Tóm tắt project

**Tên:** HCMC Air Quality Agentic RAG
**Mục tiêu:** Người dùng hỏi bằng ngôn ngữ tự nhiên (tiếng Việt hoặc tiếng Anh) về chất lượng không khí tại một địa điểm ở TP.HCM. Một LangGraph agent tự quyết định gọi tool nào, theo thứ tự nào, để:
1. Xác định tọa độ địa điểm (geocoding)
2. Lấy nồng độ chất ô nhiễm theo tọa độ (Open-Meteo Air Quality API) và tính **VN_AQI**
3. Tra khuyến nghị sức khỏe từ tài liệu chuẩn (RAG trên WHO AQG + QCVN 05:2023 + QĐ 1459/QĐ-TCMT)
4. Tổng hợp câu trả lời có căn cứ, kèm khuyến nghị hành động

**Ví dụ câu hỏi:**
> "Không khí ở phường Tân Thuận hôm nay thế nào, tôi có nên cho con ra ngoài chơi không?"

**Quyết định đã chốt (2026-10-02)** — ghi đè mọi chỗ khác trong file nếu mâu thuẫn:
| Hạng mục | Quyết định |
|---|---|
| LLM (agent, generator, judge) | **Groq API** (free tier) qua `langchain-groq`. Không dùng Claude trả phí. `LLM_MODEL=openai/gpt-oss-120b` (tool calling tốt) cho generator + agent. Groq không còn Llama chat model tại thời điểm chọn |
| Judge RAGAS | **Cerebras** (free trial) qua `ChatOpenAI` + `base_url`, `JUDGE_MODEL=qwen-3.8-27b` — **khác họ model với generator** (tránh tự chấm), cố định cho mọi variant. Giữ đủ 4 metric LLM. Lý do: đo thật judge tốn ~16K token/câu → 1 lượt V0 ≈ 375K token, vượt Groq free tier (200K token/ngày); Cerebras: header API đo 2026-10-03 cho thấy 450 RPM, 150K token/phút, 216M token/ngày (client giới hạn 30 RPM). Judge chạy `reasoning_effort=none` (tiết kiệm token); disclaimer y tế được cắt khỏi câu trả lời trước khi chấm. Không dùng nhiều tài khoản để lách quota |
| Nguồn dữ liệu AQI | **Open-Meteo Air Quality API** (dữ liệu mô hình CAMS, free, không cần key). Lý do: ngày 2026-10-02 WAQI có **0 trạm hoạt động** trong bbox TP.HCM (trạm Lãnh sự quán Mỹ ngừng gửi dữ liệu; `feed/geo` trả trạm ở Trat, Thái Lan ~480 km). Hướng mở rộng nếu cần số đo thật: OpenAQ |
| Embedding / reranker | Chạy local trên **GPU** |
| Thang AQI | **VN_AQI** theo QĐ 1459/QĐ-TCMT. Không dùng nhãn US EPA trong câu trả lời |
| Phạm vi địa lý | Bounding box **TP.HCM cũ** (trước sáp nhập 1/7/2025); tên địa danh dùng **tên mới** (sau sáp nhập, không còn cấp quận/huyện) |
| Ngôn ngữ code | Comment, docstring, README viết **tiếng Việt** |
| Thư mục gốc | `D:\LabBK\hcmc_air_quality` |
| Git | Claude Code **không commit**; người dùng tự commit |
| Quy trình | Claude Code **phải hỏi ý người dùng trước khi làm** bất kỳ việc gì (tạo/sửa file, cài package, tải dữ liệu) |

**Trạng thái:** Prototype / MVP + portfolio. Ưu tiên: chạy được với data thật, đo được bằng số.

**Nguyên tắc cốt lõi:**
- **RAG làm trước, agent làm sau.** RAG phải được build và đánh giá độc lập trước khi ghép vào agent.
- **Build theo tầng, đo sau mỗi tầng** (ablation). Không build hết rồi mới đo.
- **Không để LLM tự bịa số liệu** (tọa độ, AQI, ngưỡng nồng độ, khuyến nghị y tế). Mọi con số phải đến từ tool hoặc tài liệu.

---

## 1. Kiến trúc tổng thể

```
Người dùng (câu hỏi tự nhiên)
        │
        ▼
┌───────────────────────────────────────────┐
│ LangGraph Agent (LLM = planner)            │
│  - đọc câu hỏi + lịch sử                    │
│  - quyết định gọi tool / trả lời cuối       │
└───────┬───────────────────────────────────┘
        │ tool calls (thứ tự do agent tự quyết)
        ├──► geocode_address(address)          → Nominatim (OSM) → lat/lng
        ├──► get_air_quality(lat, lng)         → Open-Meteo → nồng độ µg/m³ → VN_AQI, PM2.5, PM10, NO2, O3, ...
        └──► retrieve_health_guideline(query)  → RAG pipeline (xem mục 4)
        │
        ▼
LLM tổng hợp → câu trả lời + khuyến nghị + nguồn trích dẫn
```

**Mapping khái niệm:**
| Khái niệm | Vị trí trong hệ thống |
|---|---|
| LLM / NLP | Hiểu câu hỏi, lập kế hoạch, sinh câu trả lời |
| Agentic AI | Vòng lặp LangGraph tự chọn & nối tool |
| GeoAI | `geocode_address`, `get_air_quality` (truy vấn theo tọa độ) |
| RAG | `retrieve_health_guideline` |

---

## 2. Tech stack

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Ngôn ngữ | Python 3.11+ | |
| Quản lý môi trường | `uv` (hoặc `venv` + `pip`) | |
| Orchestration | `langchain`, `langgraph` | |
| LLM | Groq API qua `langchain-groq` | Model cấu hình qua `.env`, không hardcode. Kiểm tra danh sách model Groq hiện hành + hỗ trợ tool calling trước khi chọn |
| Embedding | `BAAI/bge-m3` qua `langchain-huggingface` / `sentence-transformers` | Đa ngữ, tốt cho tiếng Việt. Fallback: `intfloat/multilingual-e5-base` |
| Vector store | Chroma (`langchain-chroma`), lưu local | |
| Sparse retrieval | BM25 (`rank_bm25`) qua `BM25Retriever` | |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | |
| PDF extraction | `pymupdf` (fallback `pdfplumber` cho bảng) | |
| RAG eval | `ragas` | 0.4.3 lỗi import với langchain-community 0.4.x → shim `eval/_ragas_compat.py`. Judge chạy trên Cerebras (`langchain-openai`) |
| Agent eval | `agentevals` + LangSmith | |
| Tracing | LangSmith | Bật qua env var |
| Test | `pytest` | |
| Geocoding | Nominatim (OSM) | Bắt buộc User-Agent, ≤ 1 req/s |
| Chất lượng không khí | Open-Meteo Air Quality API (`air-quality-api.open-meteo.com`) | Free, không cần key; dữ liệu mô hình CAMS (không phải trạm đo) |

> **Lưu ý cho Claude Code:** LangChain thay đổi import path thường xuyên (ví dụ `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever` có thể nằm ở `langchain`, `langchain_classic` hoặc `langchain_community` tùy version). **Luôn kiểm tra version đã cài và docs hiện hành trước khi import**, không đoán. Ghi version thực tế vào `requirements.txt` / `pyproject.toml`.

---

## 3. Cấu trúc thư mục

```
hcmc_air_quality/
├── CLAUDE.md                     # file này
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore                    # bỏ qua .env, data/raw, chroma_db, __pycache__
├── config/
│   └── settings.py               # load env, hằng số (chunk size, k, model names...)
├── data/
│   ├── raw/                      # PDF gốc (không commit nếu license không cho)
│   │   ├── who_aqg_2021_exec_summary.pdf
│   │   ├── qcvn_05_2023_btnmt.pdf
│   │   └── qd_1459_tcmt_2019.pdf     # hướng dẫn tính VN_AQI + khuyến nghị sức khỏe
│   ├── processed/                # text đã làm sạch (.md / .jsonl)
│   └── curated/
│       └── aqi_health_categories.md   # bảng VN_AQI → khuyến nghị (tổng hợp từ QĐ 1459)
├── chroma_db/                    # vector store (gitignore)
├── src/
│   ├── ingest/
│   │   ├── extract.py            # PDF → text
│   │   ├── clean.py              # làm sạch, chuẩn hóa unicode tiếng Việt
│   │   └── chunk.py              # chunking + metadata
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── vectorstore.py        # build / load Chroma
│   │   ├── retrievers.py         # dense, bm25, hybrid, multi-query, rerank
│   │   ├── prompts.py
│   │   ├── chain.py              # build_rag_chain(variant="v0".."v4")
│   │   └── tool.py               # retrieve_health_guideline (@tool)
│   ├── tools/
│   │   ├── geocode.py            # geocode_address (@tool)
│   │   ├── air_quality.py        # get_air_quality (@tool)
│   │   └── vn_aqi.py             # hàm thuần: US AQI sub-index → nồng độ → VN_AQI
│   ├── agent/
│   │   ├── prompts.py            # system prompt agent
│   │   ├── graph.py              # LangGraph StateGraph
│   │   └── run.py                # CLI chạy thử
│   └── utils/
│       └── logging.py
├── eval/
│   ├── datasets/
│   │   ├── rag_devset.jsonl      # dev 28 câu: ablation V0–V4, phân tích lỗi, chỉnh prompt
│   │   ├── rag_testset.jsonl     # test 27 câu: chỉ đo cuối Phase 1 (V0 + variant tốt nhất)
│   │   └── agent_testset.jsonl   # cùng câu + trajectory kỳ vọng
│   ├── run_rag_eval.py           # chạy RAGAS cho 1 variant
│   ├── run_ablation.py           # chạy V0→V4, xuất bảng
│   ├── run_agent_eval.py
│   └── results/                  # csv/json/md kết quả, có timestamp
├── scripts/
│   ├── build_index.py            # ingest → chunk → embed → Chroma
│   └── smoke_test.py
└── tests/
    ├── test_chunking.py
    ├── test_retrievers.py
    ├── test_tools.py
    └── test_agent_graph.py
```

---

## 4. Rule cho Claude Code (bắt buộc)

### 4.1 Quy trình làm việc
0. **Hỏi ý người dùng trước khi làm.** Trình bày việc sẽ làm → chờ đồng ý → làm → báo kết quả thật. Không tự tiện tạo/sửa file, cài package, tải dữ liệu.
1. **Làm đúng thứ tự phase** (Phase 0 → 5). Không bắt đầu phase sau khi phase trước chưa đạt "Definition of Done".
2. Trước mỗi task: đọc lại mục tương ứng trong file này, nêu ngắn gọn sẽ làm gì.
3. Sau mỗi task: chạy test/smoke test liên quan, báo kết quả thật (không báo "chạy được" nếu chưa chạy).
4. Mỗi variant RAG (V0–V4) phải được **đo xong** trước khi thêm tầng tiếp theo.
5. Khi kết quả eval xấu đi sau khi thêm 1 tầng: **ghi lại số liệu, không xóa**, báo lại cho người dùng trước khi quyết định giữ/bỏ.
6. Không tự ý mở rộng phạm vi (thêm tool, thêm nguồn data, thêm UI) nếu chưa được yêu cầu.

### 4.2 Code
- Python có type hints, docstring ngắn cho hàm public. Comment và docstring viết **tiếng Việt** (riêng docstring của tool LangChain có thể song ngữ nếu giúp LLM chọn tool tốt hơn).
- Mọi tham số có thể tinh chỉnh (chunk_size, overlap, k, trọng số hybrid, model name, top_n rerank) nằm trong `config/settings.py`, **không hardcode** rải rác.
- Secrets chỉ đọc từ `.env`. Không bao giờ commit `.env`, không in token ra log.
- Mỗi tool LangChain: input/output schema rõ ràng (Pydantic), docstring mô tả **khi nào nên dùng tool** (agent đọc docstring này để chọn tool).
- Tool phải xử lý lỗi gọn: timeout, không tìm thấy địa chỉ, API không có dữ liệu → trả về message lỗi có cấu trúc, không raise exception làm sập agent.
- Gọi API ngoài: có timeout (10s), retry tối đa 2 lần với backoff, cache kết quả geocode (dict/file) để không gọi lặp.
- Nominatim: header `User-Agent` riêng của project, tôn trọng giới hạn 1 request/giây.

### 4.3 Dữ liệu & trung thực
- Không bịa nội dung tài liệu. Nếu PDF extract lỗi (bảng vỡ, font lỗi), **báo lại** và sửa thủ công vào `data/processed/`, ghi chú đã sửa gì.
- Ngưỡng nồng độ trong QCVN/WHO phải lấy đúng từ văn bản gốc — kiểm tra lại bằng mắt sau khi extract.
- File `aqi_health_categories.md` là tài liệu tự soạn: ghi rõ ở đầu file nguồn tham chiếu (QĐ 1459/QĐ-TCMT — thang VN_AQI và khuyến nghị sức khỏe) và rằng đây là bảng tổng hợp. Nội dung khuyến nghị phải lấy từ văn bản gốc, không tự thêm.

### 4.4 Prompt
- Prompt RAG: chỉ trả lời dựa trên context; nếu context không có thông tin → nói rõ "Tài liệu không có thông tin về vấn đề này"; trích nguồn (source + section).
- System prompt agent: bắt buộc dùng tool để lấy tọa độ và AQI, không tự đoán; bắt buộc gọi `retrieve_health_guideline` trước khi đưa khuyến nghị sức khỏe; trả lời cùng ngôn ngữ với người hỏi; câu hỏi ngoài phạm vi (không liên quan không khí / không ở TP.HCM) → từ chối lịch sự.
- Thêm disclaimer ngắn: thông tin tham khảo, không thay thế tư vấn y tế.

### 4.5 Eval
- Mọi lần chạy eval lưu vào `eval/results/<YYYYMMDD-HHMM>_<variant>.json` (bộ test: `..._<variant>_test.json`) + cập nhật `eval/results/ablation.md`.
- **Tách dev/test** (quyết định 2026-10-04): `rag_devset.jsonl` (dev, 28 câu) dùng cho ablation V0–V4, phân tích lỗi, chỉnh prompt — được xem kết quả từng câu. `rag_testset.jsonl` (test, 27 câu) chỉ đo **một lần cuối Phase 1** cho V0 và variant tốt nhất (`--split test`); không xem kết quả từng câu trước lần đo đó, không chỉnh hệ thống theo bộ test. Số liệu đưa vào README/CV lấy từ bộ test.
- Ghi kèm: variant, config (chunk_size, k, weights...), model generator, model judge, số câu, thời gian chạy, latency trung bình.
- Dùng model nhỏ/rẻ làm judge khi thử nghiệm; chỉ dùng model lớn cho lần đo cuối. Ghi rõ judge model trong kết quả.
- Không "tune theo test set" một cách gian lận: không sửa câu hỏi test để điểm cao hơn.

### 4.6 Git
- **Claude Code không commit.** Người dùng tự commit sau khi task xong.
- Khi xong 1 task, Claude Code gợi ý message commit (dạng `feat(rag): add hybrid retriever (V1)`, `eval: V1 ragas results`) để người dùng dùng.
- Gợi ý người dùng commit riêng sau mỗi variant RAG để có thể checkout lại.

---

## 5. Biến môi trường (`.env.example`)

```bash
# LLM (Groq)
GROQ_API_KEY=
LLM_MODEL=                    # model Groq sinh câu trả lời / agent (phải hỗ trợ tool calling)
# Judge RAGAS (Cerebras, khác họ model với generator)
JUDGE_PROVIDER=cerebras
JUDGE_MODEL=qwen-3.8-27b
CEREBRAS_API_KEY=

# Data APIs
NOMINATIM_USER_AGENT=hcmc-aq-agent/0.1 (your-email@example.com)

# Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=hcmc-aq-agent

# RAG
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
CHROMA_DIR=./chroma_db
```

---

## 6. Nguồn dữ liệu

### 6.1 Corpus RAG
| File | Nguồn | Ghi chú |
|---|---|---|
| WHO Global Air Quality Guidelines 2021 — Executive Summary (~10 trang) | iris.who.int (handle 10665/345334) | License CC BY-NC-SA 3.0 IGO. Bắt đầu bằng bản tóm tắt. |
| WHO AQG 2021 — bản đầy đủ (273 trang) | iris.who.int (handle 10665/345329) | **Tùy chọn**, chỉ thêm nếu cần mở rộng |
| QCVN 05:2023/BTNMT | scem.gov.vn (Trung tâm Quan trắc môi trường miền Nam) — PDF free | Tránh luatvietnam / thuvienphapluat (paywall) |
| QĐ 1459/QĐ-TCMT (2019) — Hướng dẫn tính toán chỉ số chất lượng không khí Việt Nam (VN_AQI) | Tổng cục Môi trường | Nguồn chính thức cho thang VN_AQI, bảng quy đổi nồng độ → AQI, và khuyến nghị sức khỏe. **Kiểm tra còn hiệu lực / có văn bản thay thế chưa trước khi dùng** |
| `aqi_health_categories.md` | Tự soạn, tổng hợp từ QĐ 1459/QĐ-TCMT | Cầu nối giữa "chỉ số VN_AQI" và "khuyến nghị hành động" — WHO/QCVN chỉ có ngưỡng µg/m³ |

**Nội dung `aqi_health_categories.md` (tối thiểu, theo VN_AQI):**
| VN_AQI | Mức | Nhóm nhạy cảm | Người bình thường |
|---|---|---|---|
| 0–50 | Tốt | ... | ... |
| 51–100 | Trung bình | ... | ... |
| 101–150 | Kém | ... | ... |
| 151–200 | Xấu | ... | ... |
| 201–300 | Rất xấu | ... | ... |
| 301–500 | Nguy hại | ... | ... |
(Khoảng giá trị, tên mức và nội dung khuyến nghị phải đối chiếu đúng văn bản QĐ 1459.)

Kèm: định nghĩa nhóm nhạy cảm và khuyến nghị cụ thể — **chỉ lấy đúng nội dung văn bản gốc QĐ 1459** (nhóm nhạy cảm: người già, trẻ em, người mắc bệnh hô hấp, tim mạch…). **Không bổ sung** nội dung văn bản không có (vd phụ nữ mang thai, máy lọc không khí) — quyết định của người dùng ngày 2026-10-03. Câu hỏi về các nội dung đó → RAG trả "Tài liệu không có thông tin".

### 6.2 API thời gian thực
**Open-Meteo Air Quality** (không cần key)
```
GET https://air-quality-api.open-meteo.com/v1/air-quality
    ?latitude={lat}&longitude={lng}
    &hourly=pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide
    &current=pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide
    &past_days=1&timezone=Asia/Ho_Chi_Minh
```
Response quan tâm: `current.*`, `hourly.time`, `hourly.<biến>`, `hourly_units` (µg/m³), `latitude/longitude` (tâm ô lưới thực tế).
→ Tool phải trả về **nguồn (`data_source: "Open-Meteo / CAMS (mô hình)"`), tọa độ ô lưới, thời điểm dữ liệu**, để agent nói rõ đây là số liệu mô hình, không phải trạm đo.
→ Lưu ý: CAMS global có độ phân giải thô (cỡ vài chục km) → các phường gần nhau có thể ra giá trị gần như giống nhau. Ghi vào hạn chế.

**Tính VN_AQI:** Open-Meteo trả **nồng độ µg/m³** → tính VN_AQI **trực tiếp** theo công thức và bảng breakpoint của QĐ 1459 (AQI giờ / AQI ngày, đối chiếu văn bản gốc ở Phase 1; dùng `past_days` để có đủ chuỗi giờ nếu công thức cần trung bình nhiều giờ). VN_AQI = max các chỉ số phụ. Toàn bộ logic nằm trong `src/tools/vn_aqi.py`, hàm thuần, có unit test với giá trị tính tay.

**Nominatim**
```
GET https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1&countrycodes=vn
```
→ Tự động thêm ", Thành phố Hồ Chí Minh" nếu chuỗi địa chỉ không chứa tên thành phố. Kiểm tra kết quả nằm trong **TP.HCM cũ** (trước sáp nhập 1/7/2025, không gồm Bình Dương / Bà Rịa–Vũng Tàu cũ) bằng bbox + polygon ranh giới cũ; ngoài phạm vi → trả lỗi "out_of_scope". Bbox và đường dẫn polygon đặt trong `config/settings.py`.
→ Tên địa danh trong câu trả lời dùng **tên đơn vị hành chính mới** (phường/xã sau sáp nhập). Người dùng vẫn có thể hỏi bằng tên cũ ("Quận 7", "Q7", "Thủ Đức") — geocode theo địa danh rồi hiển thị tên mới nếu Nominatim trả về.

---

## 7. Kế hoạch triển khai

### PHASE 0 — Setup (≈ 2 giờ)
- [x] Tạo repo theo cấu trúc mục 3, `pyproject.toml`, `.gitignore`, `.env.example`
- [x] Cài dependencies, ghi version thực tế
- [x] `config/settings.py` load env
- [ ] Tạo LangSmith project (tùy chọn — chưa làm)
- [x] Chọn model Groq cho `LLM_MODEL` / `JUDGE_MODEL` (kiểm tra danh sách model hiện hành, tool calling, rate limit free tier) — hỏi người dùng duyệt
- [x] `scripts/smoke_test.py`: gọi thử LLM (Groq) 1 câu, Open-Meteo 1 tọa độ (trung tâm Q1 cũ: 10.7769, 106.7009), Nominatim 1 địa chỉ, kiểm tra embedding chạy trên GPU

**DoD:** smoke test chạy qua cả 4 kiểm tra.

---

### PHASE 1 — RAG (3 ngày) ⭐ trọng tâm hiện tại

Pipeline đích:
```
chunking → embedding → multi-query → hybrid search (BM25 + dense) → rerank → prompt → generator
```
Build theo variant, **đo RAGAS sau mỗi variant**:

| Variant | Thành phần | Mục đích đo |
|---|---|---|
| **V0** | dense search (top-k) → prompt → generate | Baseline |
| **V1** | V0 + hybrid (BM25 + dense, EnsembleRetriever) | Đóng góp của BM25 (thuật ngữ, mã số, đơn vị) |
| **V2** | V1 + multi-query | Đóng góp của query expansion (câu đời thường vs văn bản kỹ thuật) |
| **V3** | V2 + rerank (retrieve k=20 → rerank → top 5) | Đóng góp của cross-encoder |
| **V4** | V3 + prompt tinh chỉnh | Đóng góp của prompt engineering |

`build_rag_chain(variant: str)` phải tạo được bất kỳ variant nào từ cùng 1 codebase để ablation tái lập được.

#### Ngày 1 — Corpus + vector store → V0
**Sáng — thu thập & làm sạch**
- [x] Tải 3 PDF vào `data/raw/` (WHO AQG exec summary, QCVN 05:2023, QĐ 1459/QĐ-TCMT) — kiểm tra hiệu lực QĐ 1459
- [x] Soạn nháp `data/curated/aqi_health_categories.md` từ QĐ 1459 → người dùng duyệt
- [x] `extract.py`: PDF → text bằng pymupdf; nếu bảng vỡ → thử pdfplumber
- [x] `clean.py`: chuẩn hóa unicode NFC (tiếng Việt), bỏ header/footer lặp, sửa ngắt dòng giữa câu
- [x] **Kiểm tra bằng mắt** bảng giới hạn trong QCVN 05:2023 (bảng thông số cơ bản & độc hại) — đúng số, đúng đơn vị
- [x] Lưu text sạch vào `data/processed/`

**Chiều — chunk + embed + index**
- [x] `chunk.py`: ưu tiên split theo heading/điều khoản (1.1, 2.1, ...) cho QCVN; còn lại `RecursiveCharacterTextSplitter`
  - Mặc định: `chunk_size ≈ 600 token`, `overlap ≈ 100` (đặt trong settings)
  - Bảng: giữ nguyên 1 bảng trong 1 chunk, không cắt giữa bảng
  - Metadata: `{source, doc_title, section, language, chunk_id}`
- [x] `embeddings.py`: bge-m3 (normalize embeddings)
- [x] `vectorstore.py` + `scripts/build_index.py`: build Chroma, persist
- [x] `retrievers.py`: dense retriever
- [x] `prompts.py` + `chain.py`: V0 chain hoàn chỉnh
- [x] `tests/test_chunking.py`: không chunk rỗng, metadata đủ, bảng không bị cắt

**DoD ngày 1:**
- `retriever.invoke("AQI 160 có nên ra ngoài không")` trả về chunk liên quan (kiểm tra bằng mắt)
- V0 trả lời được 5 câu thử tay, có trích nguồn

#### Ngày 2 — Bộ test + baseline + V1
**Sáng — bộ test**
- [x] Tạo bộ câu hỏi eval, 25–30 câu, schema (bộ đầu tiên nay là `rag_devset.jsonl`, xem mục 4.5):
```json
{
  "id": "q001",
  "group": "threshold | aqi_advice | reasoning | out_of_scope",
  "language": "vi | en",
  "question": "...",
  "ground_truth": "...",
  "reference_contexts": ["đoạn tài liệu lẽ ra phải tìm được"],
  "reference_source": "qcvn_05_2023 | who_aqg_2021 | qd_1459_vn_aqi | aqi_categories | none"
}
```
- [x] Phân bổ:
| Nhóm | Số câu | Ví dụ |
|---|---|---|
| `threshold` — tra ngưỡng trực tiếp | ~10 | "Giới hạn PM2.5 trung bình 24 giờ theo QCVN 05:2023 là bao nhiêu?" |
| `aqi_advice` — khuyến nghị theo mức VN_AQI | ~8 | "AQI 180 thì người bị hen suyễn nên làm gì?" |
| `reasoning` — tổng hợp/so sánh | ~5 | "Ngưỡng PM2.5 của WHO và QCVN khác nhau thế nào?" |
| `out_of_scope` — bẫy | ~5 | "Máy lọc không khí loại nào tốt nhất?" → phải nói không có thông tin |
- [x] Trộn câu tiếng Việt và tiếng Anh, trộn văn phong đời thường và văn phong kỹ thuật
- [x] **Người dùng duyệt bộ test** trước khi chạy eval (Claude Code soạn nháp, người dùng xác nhận ground truth)
- [x] Soạn bộ test riêng `rag_testset.jsonl` (27 câu t001–t027, cùng schema và phân bổ) — người dùng duyệt 2026-10-05

**Chiều — eval baseline + V1**
- [x] `eval/run_rag_eval.py --variant v0`: chạy RAGAS
  - Metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`
  - Đo thêm: latency trung bình/câu, tỉ lệ từ chối đúng ở nhóm `out_of_scope`
- [x] Ghi baseline vào `eval/results/ablation.md`
- [x] `retrievers.py`: BM25 retriever (trên cùng tập chunk) + hybrid ensemble (trọng số mặc định 0.5/0.5)
- [x] Chạy eval V1, cập nhật bảng

**DoD ngày 2:** bộ test được duyệt; có số V0 và V1.

#### Ngày 3 — V2, V3, V4 + tổng hợp
- [x] V2: multi-query (sinh 3 biến thể câu hỏi, gộp kết quả, khử trùng lặp) → eval
- [x] V3: retrieve rộng k=20 → cross-encoder bge-reranker-v2-m3 → top 5 → eval
- [x] V4: tinh chỉnh prompt dựa trên lỗi quan sát được → eval (2 lần: 2026-10-05, 2026-10-06; `ablation.md` ghi trung bình)
- [x] Phân tích lỗi: liệt kê các câu điểm thấp nhất ở variant tốt nhất, phân loại nguyên nhân (`eval/results/error_analysis.md`, V0–V4)
- [x] Chốt variant tốt nhất làm mặc định cho tool — **V4** (2026-10-06, `settings.rag.tool_variant`)
- [ ] Đo bộ test (`--split test`) cho V0 và variant tốt nhất, ghi vào bảng test trong `ablation.md` (V0 xong 2026-10-06; còn V4)
- [x] `src/rag/tool.py`: `retrieve_health_guideline(query: str)` (@tool), trả về `{answer, found, sources, variant}` hoặc `{error, message}`; test `tests/test_rag_tool.py`

**Bảng chẩn đoán khi metric thấp:**
| Metric thấp | Nguyên nhân thường gặp | Hướng sửa |
|---|---|---|
| Context recall | Chunk quá nhỏ/to; embedding kém tiếng Việt; bảng bị cắt | Đổi chunk size, split theo điều khoản, thử embedding khác |
| Context precision | Lấy nhiều chunk rác | Giảm k, thêm rerank, chỉnh trọng số hybrid |
| Faithfulness | LLM bịa ngoài context | Siết prompt, yêu cầu trích dẫn |
| Answer relevancy | Trả lời lan man | Prompt yêu cầu trả lời trực tiếp, ngắn |

**Ưu tiên khi thiếu thời gian:** cắt V2 (multi-query) trước; giữ hybrid + rerank.

**DoD Phase 1:**
- Bảng ablation V0→V4 đầy đủ trong `eval/results/ablation.md`
- Tool `retrieve_health_guideline` chạy độc lập, có test

**Mẫu `ablation.md`:**
| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | Latency (s) | Ghi chú |
|---|---|---|---|---|---|---|---|
| V0 dense | | | | | | | |
| V1 +hybrid | | | | | | | |
| V2 +multi-query | | | | | | | |
| V3 +rerank | | | | | | | |
| V4 +prompt | | | | | | | |

---

### PHASE 2 — Geo tools (≈ 0.5–1 ngày)
> Làm trước khi đóng Phase 1 — quyết định của người dùng 2026-10-05 (Phase 2 không phụ thuộc RAG).

- [x] `tools/geocode.py` — `geocode_address(address: str)`
  - Output: `{lat, lng, display_name, ward, query, source}` hoặc `{error, message}` (`ward` = phường/xã mới thay cho `district`, vì sau sáp nhập không còn cấp quận/huyện)
  - Cache, rate limit, kiểm tra phạm vi TP.HCM cũ: bbox → **polygon** (`data/geo/hcmc_old_boundary.geojson`, geoBoundaries gbHumanitarian — Chính phủ VN qua HDX/OCHA, 2020, CC BY 3.0 IGO). Bbox một mình không đủ: trùm cả Thủ Dầu Một; OSM đã ghi Bình Dương cũ là "Thành phố Hồ Chí Minh". Bản gbOpen (vẽ lại từ Wikipedia) bị loại vì lệch ~8 km
  - Địa danh cũ không kèm phường (vd node lịch sử "Quận 7") → reverse geocode lấy phường/xã mới
- [x] `tools/vn_aqi.py` — hàm thuần tính VN_AQI từ nồng độ µg/m³ theo QĐ 1459 (xem mục 6.2), unit test với giá trị tính tay (ví dụ mục 2.3 của văn bản)
- [x] `tools/air_quality.py` — `get_air_quality(lat: float, lng: float)`
  - Output: `{vn_aqi, category, category_color, dominant_pollutant, sub_indices, pm25, pm10, no2, o3, so2, co, pm25_nowcast, pm10_nowcast, unit, measured_at, data_source, grid_lat, grid_lng, method, note}` hoặc `{error, message}`
  - VN_AQI **giờ** (PM dùng Nowcast 12 giờ), bỏ các giờ dự báo sau `current.time`
  - Tính `category` theo thang VN_AQI (hàm thuần, có unit test)
  - `note`: luôn nhắc đây là dữ liệu mô hình CAMS độ phân giải thô, không phải trạm đo
- [x] `tests/test_tools.py`: test với mock response (không gọi API thật trong unit test) + integration test đánh dấu riêng (`pytest -m integration`; mặc định bị bỏ qua qua `addopts`)
- [x] Sanity check thật (2026-10-05): Q1 cũ, Q7 cũ, Thủ Đức cũ, Bình Tân cũ, Củ Chi cũ (tên cũ lẫn tên phường/xã mới) đều trả đúng format; Thủ Dầu Một bị từ chối `out_of_scope`

**DoD:** 2 tool trả đúng format cho 5 địa điểm; lỗi được xử lý không làm sập.

---

### PHASE 3 — Agent LangGraph (≈ 1 ngày)
- [x] `agent/graph.py`: StateGraph với node `agent` (LLM bind_tools) + node `tools` (ToolNode), cạnh điều kiện: có tool call → tools, không → END
- [x] Giới hạn số bước (recursion limit ~ 8) để tránh loop
- [x] Memory nhiều lượt (checkpointer) — hỏi tiếp "vậy có nên mở cửa sổ không?" không cần nhắc lại địa chỉ
- [x] `agent/prompts.py`: system prompt theo mục 4.4 (kèm thời điểm hiện tại — lỗi q001)
- [x] `agent/run.py`: CLI chat, in ra trace tool calls (tên tool + args) để debug
- [x] `tests/test_agent_graph.py`: với LLM giả/mock, kiểm tra graph route đúng
- [ ] Chạy thử 10 câu, xem trace trên LangSmith

**Trajectory kỳ vọng điển hình:**
| Loại câu hỏi | Trajectory |
|---|---|
| Hỏi AQI + khuyến nghị tại địa điểm | geocode → get_air_quality → retrieve_health_guideline → answer |
| Chỉ hỏi AQI tại địa điểm | geocode → get_air_quality → answer |
| Chỉ hỏi kiến thức (ngưỡng QCVN...) | retrieve_health_guideline → answer |
| Ngoài phạm vi | answer (từ chối), không gọi tool |
| Câu hỏi tiếp theo cùng địa điểm | dùng lại tọa độ từ memory, không geocode lại |

**DoD:** agent trả lời đúng luồng cho 10 câu thử, không bịa số liệu.

---

### PHASE 4 — Agent evaluation (≈ 1 ngày)
- [ ] `eval/datasets/agent_testset.jsonl`: dùng lại câu hỏi phù hợp + thêm câu có địa điểm, thêm cột:
```json
{
  "id": "a001",
  "question": "...",
  "expected_trajectory": ["geocode_address", "get_air_quality", "retrieve_health_guideline"],
  "expected_args_hints": {"geocode_address": "Quận 7"},
  "category": "full | aqi_only | knowledge_only | out_of_scope | follow_up | ambiguous"
}
```
- [ ] Thêm câu khó: địa điểm mơ hồ ("chỗ tôi"), địa điểm ngoài TP.HCM, follow-up nhiều lượt
- [ ] `eval/run_agent_eval.py` với `agentevals`:
  - Trajectory match (strict / unordered / superset tùy loại)
  - LLM-as-judge trajectory (không cần reference)
- [ ] Metrics: tool selection accuracy, trajectory match rate, số bước trung bình vs tối thiểu, tỉ lệ loop, tỉ lệ từ chối đúng, latency end-to-end
- [ ] Lưu kết quả + bảng tổng hợp vào `eval/results/agent_eval.md`
- [ ] Phân tích lỗi: lỗi do chọn tool sai vs do tool trả kết quả sai vs do RAG

**DoD:** có bảng số liệu agent eval + danh sách lỗi đã phân loại.

---

### PHASE 5 — Hoàn thiện (tùy chọn)
- [ ] UI đơn giản (Streamlit hoặc Gradio): chat + bản đồ Leaflet/folium hiển thị điểm hỏi và giá trị VN_AQI
- [ ] README: kiến trúc, cách chạy, bảng kết quả ablation + agent eval, hạn chế đã biết
- [ ] Demo GIF / video ngắn
- [ ] Cập nhật CV: đổi "prototyping" → "built", thêm số liệu thật

---

## 8. Timeline tổng

| Phase | Thời lượng | Trạng thái |
|---|---|---|
| 0 Setup | ~2 giờ | ✅ |
| 1 RAG (V0→V4) | 3 ngày | 🟨 |
| 2 Geo tools | 0.5–1 ngày | ✅ |
| 3 Agent | 1 ngày | 🟨 |
| 4 Agent eval | 1 ngày | ⬜ |
| 5 Hoàn thiện | tùy chọn | ⬜ |

Claude Code: cập nhật cột trạng thái (⬜ → 🟨 đang làm → ✅ xong) khi hoàn thành phase.

---

## 9. Rủi ro đã biết & cách xử lý

| Rủi ro | Xử lý |
|---|---|
| PDF QCVN extract lỗi bảng/font | pdfplumber hoặc sửa tay vào `data/processed/`, ghi chú |
| Corpus nhỏ → BM25/rerank cải thiện ít | Vẫn ghi kết quả thật; đó cũng là một phát hiện hợp lệ |
| WAQI không có trạm hoạt động ở TP.HCM (đã xảy ra 2026-10-02) | Đã chuyển sang Open-Meteo; OpenAQ là hướng mở rộng nếu cần số đo thật |
| Dữ liệu Open-Meteo là mô hình (CAMS), có thể lệch so với đo thực tế; độ phân giải thô | Gắn `data_source` + `note`, agent nói rõ trong câu trả lời; ghi vào hạn chế README |
| Open-Meteo không trả dữ liệu / timeout | Trả lỗi có cấu trúc, agent báo không lấy được số liệu, không tự đoán |
| Nominatim không hiểu địa chỉ tiếng Việt không dấu / viết tắt ("Q7") | Chuẩn hóa ("Q7" → "Quận 7"), thêm hậu tố thành phố |
| Tên đơn vị hành chính thay đổi sau sáp nhập (1/7/2025) | Nhận cả tên cũ lẫn mới khi hỏi; trả lời bằng tên mới; dữ liệu OSM có thể chưa cập nhật hết → ghi vào hạn chế |
| QĐ 1459 hết hiệu lực / bị thay thế | Kiểm tra trước khi ingest; nếu có văn bản mới thì báo người dùng |
| Groq free tier giới hạn rate (30 RPM, 8K TPM, 200K token/ngày mỗi model) | Retry có backoff khi 429; judge RAGAS chuyển sang Cerebras; chạy subset khi debug |
| Cerebras free trial: giới hạn thực tế (header API, 2026-10-03) 450 RPM / 150K TPM / 216M token/ngày; có thể là credit giới hạn thời gian (nguồn bên thứ ba nói 5 USD / 30 ngày, chưa xác minh) | Rate limiter phía client 30 RPM (429 → giảm 20); ghi token usage mỗi lượt; nếu trial hết → báo người dùng trước khi đổi judge (đổi judge = phải chấm lại mọi variant) |
| Kết quả RAGAS dao động giữa các lần chạy | Chạy ≥ 2 lần cho variant cuối, báo trung bình |

---

## 10. Ngoài phạm vi (không làm trong MVP)
- Dự báo AQI (forecasting model)
- Dữ liệu ảnh vệ tinh, mô hình CV
- Ngập lụt, thiên tai khác
- Triển khai production, auth, multi-user
- Fine-tune embedding / LLM

---

## 11. Mô tả CV hiện tại
> Assigned to explore **GeoAI**: prototyping an **agentic RAG** system for Ho Chi Minh City air quality — **LangGraph** agent autonomously selecting and chaining OSM/Open-Meteo spatial tools and **hybrid retrieval** over WHO and QCVN 05:2023 guidelines, with **RAGAS** and trajectory-based evaluation planned.

Cập nhật sau Phase 4 bằng số liệu thật (vd: "improving context recall from X to Y across five pipeline variants").
