# Hình 1 — Hướng dẫn tổng quan về Workflow (Phiên bản Hội nghị A*)

> **Mục đích**: Hướng dẫn toàn diện để vẽ lại hình kiến trúc hệ thống ADAPT-Rehab phục vụ cho việc **nộp bài hội nghị xếp hạng A\*** (CVPR / AAAI / CHI / NeurIPS / IJCAI / EMNLP — các hội nghị được liệt kê trong `paper/sections/abstract.tex` và `CLAUDE.md`).
>
> **Đối tượng độc giả**: bạn (tác giả) khi bắt tay vào vẽ hình trên draw.io / Figma / TikZ.
>
> **Các tài liệu/mã nguồn liên quan**: `paper/sections/methodology.tex` §III.A, `paper/sections/introduction.tex` §I, `paper/figures/fig1_architecture.mmd` (bản mẫu mermaid hiện tại), `main_v3.py` (mã nguồn runtime định nghĩa các thành phần thực tế).

---

## 0. Tại sao cần vẽ lại Hình mới cho Hội nghị A*?

Các hội nghị xếp hạng A\* khác với IEEE ATC (mục tiêu trước đó) ở ba khía cạnh dưới đây, làm thay đổi cách thiết kế Hình 1:

| Khía cạnh | IEEE ATC (Hạng B) | A* (CVPR/AAAI/CHI/NeurIPS) |
|---|---|---|
| Chiều rộng trang | Hai cột, 8.5" — Hình 1 nằm trong `figure*` (toàn bộ chiều rộng) | NeurIPS: một cột 5.5"; CVPR: hai cột 6.875"; AAAI: hai cột 6.75". Hình 1 ở các hội nghị A\* hầu như luôn được đặt ở **một cột duy nhất phía trên cùng của trang 1 hoặc trang 2** (phần đầu trang nhìn thấy ngay), chứ không phải là một `figure*` rộng. |
| Phong cách trực quan | Mang tính chức năng, "các hộp và mũi tên" | Sạch sẽ hơn, giống phong cách poster hơn, nhiều khoảng trắng, 1-2 màu nhấn (accent) mạnh mẽ, 4-5 màu trung tính dịu hơn. Tiết chế, không dùng kiểu "cầu vồng nhiều màu". |
| Khoảng cách đọc được | Bản in đen trắng (B&W) ở mức chấp nhận được | Cả bản màu và bản đen trắng đều phải hoạt động tốt, nhưng người phản biện thường xem trên màn hình — màu sắc bão hòa sẽ hiển thị tốt, trong khi màu pastel trông sẽ bị mờ nhạt khi phóng to 100%. |
| Giới hạn mật độ | Có thể chứa 18 thành phần | Hội nghị A\* từ chối các hộp dạng "bức tường văn bản". Giới hạn tối đa là 4-6 thành phần cho mỗi lớp. Các chi tiết sâu hơn nên được ẩn bên trong các hình phụ (sub-figure). |
| Phong cách thiết kế | Phẳng theo chuẩn IEEE | Hiện đại: bo góc nhẹ (4–6 px), nét viền (stroke) 1.25–1.5 px, **không đổ bóng**, **không dùng dải màu (gradient)** (phong cách CVPR/AAAI), không dùng hiệu ứng 3D. NeurIPS/EMNLP cho phép sử dụng gradient rất nhẹ. |
| Giọng điệu của chú thích (Caption) | Khô khan: "mô tả quy trình năm lớp" | Mang tính quảng bá — một câu nhấn mạnh về **đóng góp khoa học** (lý do *tại sao*), sau đó mới mô tả cấu trúc cơ học. |

Cụ thể: hình hiện tại của IEEE ATC **quá dày đặc** (3 thành phần trong Input, 3 trong Perception, 5 trong Analysis, 4 trong Intelligence, 3 trong Output = 18 hộp kích thước 200×70 px trong một khung vẽ 1600×900) và **quá nhiều màu pastel** đối với tiêu chuẩn của hội nghị A*. Hướng dẫn này sẽ tạo ra một phiên bản tinh gọn hơn chỉ với **13 hộp**, được tổ chức sao cho mắt người đọc sẽ tập trung vào **điểm mới/đóng góp chính** (3D pose, LLM coaching, pain/fatigue) thay vì các phần thông thường (webcam, TTS, dashboard).

---

## 1. Bố cục chuẩn hội nghị A* — Kỳ vọng từ Người phản biện

Hình "tổng quan hệ thống" chuẩn mực cho một hệ thống AI đa phương thức tại các hội nghị A* tuân theo **bố cục ba vùng từ trái qua phải** cùng với một cung phản hồi lớn:

```
              ┌─────────────────────────────────────────────────────────────┐
              │                       USER / ENVIRONMENT                   │   (top strip, optional)
              └─────────────────────────────────────────────────────────────┘
                                       │
   ╔════════════╗   ╔════════════╗   ╔════════════╗   ╔════════════╗   ╔════════════╗
   ║  SENSING   ║──▶║ PERCEPTION ║──▶║  ANALYSIS  ║──▶║INTELLIGENCE║──▶║ INTERFACE  ║
   ║  (Input)   ║   ║            ║   ║            ║   ║            ║   ║  (Output)  ║
   ╚════════════╝   ╚════════════╝   ╚════════════╝   ╚════════════╝   ╚════════════╝
            └─────────────────── feedback arc (dashed) ──────────────────┘
```

Năm cột. Các cột này **không phải là "các lớp" (layers)** theo nghĩa kỹ thuật — chúng là **các giai đoạn của câu chuyện** tương ứng với thứ tự đóng góp của bài báo trong phần §I (Mở đầu). Cung phản hồi (feedback arc) thể hiện khía cạnh an toàn: Đầu ra (cảm biến phát hiện đau/mệt mỏi - pain/fatigue sensor) ghi ngược lại về phía Trí tuệ (Intelligence) để huấn luyện viên (coach) có thể điều chỉnh.

**Các quy ước được sử dụng tại các hội nghị A\***:
- Các cột giai đoạn có **cùng chiều rộng, cùng chiều cao và khoảng cách dọc bằng nhau** (tạo nhịp điệu trực quan).
- Các mũi tên là **nét đơn (single-stroke), 1.5–2 px, chỉ có một đầu mũi tên**, không trang trí.
- Mỗi cột có **một màu nhấn** (xem phần §3) và **các hộp bên trong màu trắng**.
- Vòng lặp phản hồi (feedback loop) được vẽ dưới dạng một cung nét đứt duy nhất ở phía dưới — tuyệt đối không vẽ nhiều mũi tên đan chéo nhau.
- Một **chú giải (legend)** nhỏ nằm ở góc (6–8 pt) giải thích ý nghĩa của mũi tên / biểu tượng.
- Một **dải biểu tượng ở trên cùng** thể hiện các phương thức (camera, mic, người, loa) thường được hoan nghênh tại CHI; nên lược bỏ tại CVPR/AAAI/NeurIPS.

---

## 2. Các khối bắt buộc phải vẽ (phiên bản chuẩn 13 hộp)

Hình 1 chuẩn hội nghị A* phải thể hiện **điểm gì mới** ở cấp độ cột và **phần kỹ thuật** nằm bên trong. Mỗi cột có **2-3 hộp**. Cách đánh số bên dưới tương ứng với tham chiếu chú thích trong LaTeX (Fig. 1 trong `methodology.tex`).

### Cột 1 — SENSING (hay còn gọi là "Input" trong §III.A)
*Vai trò*: nguồn dữ liệu mà người dùng không thể can thiệp. **2 hộp** — không nên lãng phí không gian cho hồ sơ người dùng (user profile), hãy gộp nó vào một dải biểu tượng nhỏ ở bên cạnh.

| # | Nhãn hộp | Nhãn phụ (Sub-label - 10 pt) | Ghi chú |
|---|---|---|---|
| 1.1 | **RGB + Depth Camera** | "30 FPS, 1080p" | Đủ tổng quát để bao quát webcam + Kinect trong phần đánh giá. Thông số "depth" được đề cập trong hạn chế 1 ở phần §I như một thứ chúng ta *tránh* yêu cầu. |
| 1.2 | **Microphone** | "Vietnamese ASR" | Chúng ta bỏ từ "Whisper" khỏi hộp — tên mô hình thuộc về cột *perception*, không phải đầu vào (input). |
| 1.3 | **User Profile** *(tùy chọn, gộp vào dải biểu tượng)* | "Age · ROM · Conditions" | Vẽ dưới dạng một biểu tượng `JSON` nhỏ cạnh camera, không vẽ thành hộp riêng. |

### Cột 2 — PERCEPTION
*Vai trò*: chuyển đổi các tín hiệu thô thành các dự đoán có cấu trúc. **3 hộp**, và đây là các hộp có *thiết kế mới lạ* — hãy tô màu nhấn cho chúng.

| # | Nhãn hộp | Nhãn phụ (10 pt) | Tại sao nó ở đây |
|---|---|---|---|
| 2.1 | **RTMW3D-L** | "Whole-body 3D, 133 KP" | Đóng góp số #2 từ §I. Hậu tố "L" rất quan trọng — đó là biến thể thu gọn (lightweight). |
| 2.2 | **OpenFace 3.0** | "8 AU + 8 Emotion" | Đây là công cụ chạy đường truyền (pipeline) khuôn mặt của *thân thể*. |
| 2.3 | **Whisper ASR** | "vi-VN, large-v3" | Điểm mới trong phiên bản A*: một hộp nhỏ, không phải cả cột lớn. Hiển thị nó ở đây như một mô-đun nhận thức giọng nói (perception module), sau đó nó sẽ truyền dữ liệu sang cột Intelligence. |

> **Mẹo**: giữ cho đường truyền phụ **body** (RTMW3D) và đường truyền phụ **face** (OpenFace) có sự khác biệt trực quan trong cột này — chỉ cần một đường phân cách dọc mỏng là đủ. Người phản biện sẽ hiểu đây là "hai phương thức, chung một cột".

### Cột 3 — ANALYSIS
*Vai trò*: từ các dự đoán thô chuyển thành các đặc trưng lâm sàng. **3 hộp** — gộp 5 hộp của bản ATC cũ thành 3 nhóm khái niệm.

| # | Nhãn hộp | Nhãn phụ (10 pt) | Những phần được tích hợp từ phiên bản ATC |
|---|---|---|---|
| 3.1 | **Quaternion Kinematics** | "8 bilateral joints · ISB" | Gộp "Quaternion Kinematics" + "SPARC + LDLJ" (độ mượt - smoothness được tính toán *từ* các góc, do đó nó thuộc về phần động học kinematics) |
| 3.2 | **Constrained DTW** | "Sakoe-Chiba band" | Giữ nguyên như cũ |
| 3.3 | **Compensation + Fatigue** | "Temporal LSTM" | Gộp "Compensation + Fatigue" + "6-Dim Scoring". Phần tính điểm (Scoring) nằm ở sau phân tích (downstream); hãy chuyển nó thành một nhãn phụ nhỏ (badge) bên trong cột **Intelligence**. |

### Cột 4 — INTELLIGENCE
*Vai trò*: từ các đặc trưng chuyển thành *việc cần làm và lời cần nói*. Đây là cột mà người phản biện sẽ nhìn chăm chú nhất. **3 hộp**, cộng thêm một chip tính điểm nhỏ.

| # | Nhãn hộp | Nhãn phụ (10 pt) | Ghi chú |
|---|---|---|---|
| 4.1 | **LLM Coach** | "GPT-4o / Claude" | Đóng góp khoa học số #4 từ §I. **Hãy làm cho hộp này cao nhất** (hoặc thêm nhãn phụ "Reasoning" nhỏ) để biểu thị đây là bộ não của hệ thống. |
| 4.2 | **RAG + Safety** | "Clinical KB · Guardrails" | Điểm mới ở đây là cơ sở tri thức lâm sàng bằng *tiếng Việt*. |
| 4.3 | **Edge-TTS** | "vi-VN-HoaiMyNeural" | Vẽ như một hộp nhỏ ở *dưới cùng* của cột, hướng trực quan về phía cột Output. |
| 4.4 (chip) | **Score (6-D)** | "ROM · Stability · Flow · …" | Thể hiện dưới dạng một chip bo góc nhỏ ở cạnh hộp 4.1 — *không* vẽ thành một hộp đầy đủ. Giúp tiết kiệm diện tích vẽ. |

### Cột 5 — INTERFACE
*Vai trò*: những gì người dùng lớn tuổi thực sự nhìn và nghe thấy. **2 hộp** — bảng điều khiển (dashboard) chỉ là phần kỹ thuật thông thường, không phải đóng góp nghiên cứu.

| # | Nhãn hộp | Nhãn phụ (10 pt) | Ghi chú |
|---|---|---|---|
| 5.1 | **Visual Coach** | "Skeleton + ROM arcs" | Phần mà người phản biện sẽ nhìn thấy trong video demo của bạn. |
| 5.2 | **Voice Feedback** | "Vietnamese instructions" | Đủ tổng quát để nhận đầu ra từ TTS. |

### Cung phản hồi - Feedback Arc (câu chuyện về sự an toàn)
`Một mũi tên nét đứt đơn` từ **Cột 4 → Cột 3 → Cột 4** là **sai** (tạo các vòng lặp bên trong hình). Hãy sử dụng **một** trong hai mẫu thiết kế chuẩn hội nghị A* sau:

- **Mẫu A (khuyến nghị cho hội nghị A*)**: một mũi tên nét đứt đơn từ phía dưới **Cột 5** (Interface), uốn cong ngược lại bên dưới cả năm cột, và kết thúc tại **Cột 4** (Intelligence). Nhãn: *"Pain / fatigue → adapt coaching"*. Mẫu này tương tự như mũi tên "phản hồi thời gian thực" đã có sẵn trong `fig1_architecture.mmd` dòng 67.
- **Mẫu B (phong cách hội nghị CHI)**: một vòng lặp màu **đỏ** riêng biệt từ chip nhỏ "Pain / Fatigue" ở phía dưới bên phải của cột Perception, vòng lên và quay lại LLM Coach. Hai chip, một cung vẽ. Phù hợp hơn nếu bạn muốn *nhấn mạnh* đóng góp về mặt an toàn.

### Dải trên cùng tùy chọn (Optional Top Strip)
*Chỉ thêm vào nếu nộp cho hội nghị CHI hoặc các hội nghị thuộc lĩnh vực tương tác người-máy (HCI) khác.* Một dải mỏng chạy dọc phía trên cùng chứa ba biểu tượng:
- 👤 *Người dùng lớn tuổi (60+)*
- 🏥 *Hướng dẫn lâm sàng (nguồn RAG)*
- 📊 *Nhật ký phiên tập → bác sĩ*

Đối với các hội nghị CVPR/AAAI/NeurIPS, **hãy lược bỏ phần này** — các hội nghị về Thị giác máy tính/Trí tuệ nhân tạo (CV/AI) hạng A* ưa chuộng các hình ảnh có tính độc lập cao, tự giải thích được nội dung.

---

## 3. Bảng màu (An toàn cho Hội nghị A*, Bản in và Đen trắng)

### 3.1 Quy tắc tối đa hai màu nhấn (Two-Accent Rule)

Các hình ảnh trong hội nghị A* sử dụng **tối đa hai màu nhấn** cộng với dải màu trung tính. ADAPT-Rehab có hai đóng góp khoa học xứng đáng được tô màu nhấn:
- **Màu xanh dương** dành cho **đóng góp về mặt nhận thức/pose 3D** (Đóng góp #2 từ phần §I)
- **Màu đỏ/cam ấm** dành cho **đóng góp về mặt an toàn/LLM-coaching** (Đóng góp #4 — vì màu đỏ là ký hiệu chung cho sự "an toàn", và vòng lặp phản hồi của bạn liên quan đến đau/mệt mỏi)

Mọi thành phần khác sẽ sử dụng **dải màu xám trung tính**. Không dùng màu xanh lá, màu tím hay màu pastel.

### 3.2 Bảng màu chính xác

| Vai trò | Hex | sRGB | CMYK (cho bản in) | Cách dùng |
|---|---|---|---|---|
| **Màu nhấn chính — Pose 3D** | `#1F4E79` | xanh dương đậm | 100/70/10/40 | Viền cột 2 + thanh tiêu đề, nét viền hộp RTMW3D (3 px) |
| **Màu tô của màu nhấn chính** | `#D9E2F3` | tông xanh nhạt | 15/5/0/0 | Màu nền cột 2, dải tiêu đề cột "Perception" |
| **Màu nhấn phụ — An toàn** | `#C0392B` | đỏ ấm | 15/95/90/5 | Nét vẽ cung phản hồi, nét viền hộp "RAG + Safety" (2 px) |
| **Màu tô của màu nhấn phụ** | `#FADBD8` | tông đỏ nhạt | 0/30/15/0 | Chỉ dùng để tô nền cho hộp RAG + Safety (không tô cả cột) |
| **Màu trung tính — Kỹ thuật** | `#566573` | xám vừa | 60/45/40/10 | Viền các cột khác, nét vẽ mũi tên |
| **Màu trung tính — Bề mặt nền** | `#F4F6F7` | xám rất nhạt | 2/1/1/0 | Nền của tất cả các cột khác |
| **Màu trung tính — Hộp** | `#FFFFFF` | trắng | 0/0/0/0 | Màu tô bên trong các hộp thành phần |
| **Màu trung tính — Viền hộp** | `#2C3E50` | xanh gần đen | 90/65/45/55 | Viền các hộp thành phần (1.25 px) |
| **Màu trung tính — Chữ chính** | `#1B2631` | đen gần tuyệt đối | 90/70/50/60 | Toàn bộ tiêu đề, văn bản chính |
| **Màu trung tính — Chữ phụ** | `#566573` | xám vừa | 60/45/40/10 | Nhãn phụ, nhãn mũi tên |

### 3.3 Tại sao nên chọn bảng màu này

- **Xanh dương + Đỏ** là hệ thống hai màu nhấn phổ biến nhất ở các hội nghị A* (NeurIPS, CVPR, CHI đều sử dụng hệ màu này). Nó giúp người đọc dễ dàng nhận biết ngay từ cái nhìn đầu tiên.
- `#1F4E79` (màu xanh dương) **đủ tối để in ra bản đen trắng rõ nét** (độ sáng luminance ≈ 25%). Các màu xanh nhạt hơn như `#3498DB` trông sẽ giống hệt nhau khi chuyển sang thang màu xám.
- `#C0392B` (màu đỏ) có **độ sáng luminance ≈ 35%**, giúp phân biệt rõ ràng với màu xanh dương trong bản in đen trắng.
- Tất cả các màu xám đều có **khoảng cách độ sáng ≥ 15%**, vì vậy chúng vẫn dễ đọc khi bài báo được photocopy.
- Bảng màu này vượt qua kiểm tra **an toàn với người mù màu của Wong (2011)** đối với mù màu xanh lục (deuteranopia) và mù màu đỏ (protanopia) — xanh dương/đỏ là sự kết hợp hai màu an toàn nhất.

### 3.4 Các màu bị cấm

- ❌ `#D6EAF8`, `#D5F5E3`, `#FEF9E7`, `#E8DAEF`, `#FDEBD0` (các tông màu pastel trong phiên bản ATC cũ) — chúng trông giống sách giáo khoa cho trẻ em tại các hội nghị A*.
- ❌ `#000000` đen tuyền cho viền hộp — quá thô cứng; hãy dùng mã `#2C3E50` để tạo cảm giác ấm áp hơn.
- ❌ Bất kỳ hiệu ứng chuyển màu (gradient) hay đổ bóng (drop shadow) nào — thiết kế phẳng (flat design) là quy chuẩn của hội nghị A*.
- ❌ Màu vàng làm màu nhấn — sẽ không thể nhìn thấy trong bản in đen trắng và không tốt cho người đọc bị mù màu.

---

## 4. Kiểu chữ (Typography)

Các hình ảnh trong hội nghị A* tuân theo các quy tắc thiết kế kiểu chữ sau (đồng bộ với hướng dẫn định dạng của NeurIPS / CVPR / AAAI):

| Phần tử | Phông chữ | Kích thước | Độ đậm (Weight) | Màu sắc |
|---|---|---|---|---|
| Tiêu đề cột (Nhãn "PERCEPTION") | **Helvetica** hoặc **Arial** | **14 pt** | **Bold** | `#1B2631` |
| Tiêu đề hộp thành phần | Giống trên | **12 pt** | **Bold** | `#1B2631` |
| Nhãn phụ hộp thành phần | Giống trên | **9.5 pt** | Regular | `#566573` |
| Nhãn mũi tên | Giống trên | **9 pt** | Regular | `#1B2631` |
| Nhãn cung phản hồi | Giống trên | **9 pt** | *Italic* | `#C0392B` |
| Chú giải (nếu dùng) | Giống trên | **8 pt** | Regular | `#566573` |
| Số thứ tự hình (Ví dụ "1") — *không vẽ trực tiếp vào hình* | — | — | — | — |

**Quy tắc chung**:
- Toàn bộ văn bản phải đạt kích thước **≥ 8 pt ở kích thước in cuối cùng**. Nếu khung vẽ (canvas) của bạn rộng 1600 px và hình vẽ hiển thị ở kích thước 5.5" (cột đơn NeurIPS), khi đó 1 pt ≈ 22 px. Vì vậy, 8 pt ≈ 176 px trên khung vẽ. **Kích thước tối thiểu này lớn hơn nhiều so với phiên bản IEEE** — người phản biện hội nghị A* thường phóng to hình để xem chi tiết.
- Sử dụng **một phông chữ duy nhất xuyên suốt hình**. Không kết hợp Arial và Times. Helvetica/Arial is the safe default for figures.
- Nhãn phụ thành phần được in *nghiêng (italics)* ở một số bài báo AAAI, và in *thường (regular)* ở một số bài báo khác. Chúng tôi khuyên dùng chữ thường + kích thước nhỏ hơn (để tiết kiệm mực).

---

## 5. Hình học & Khoảng cách (Geometry & Spacing)

Một hình vẽ cột đơn chuẩn hội nghị A* có kích thước khoảng ~5.5" rộng × 2.5" cao (NeurIPS) hoặc ~6.75" rộng × 2.8" cao (AAAI). Ở mật độ 300 dpi, tương ứng với:

| Biến thể khung vẽ (Canvas) | Kích thước pixel (Rộng × Cao) | Phù hợp nhất cho |
|---|---|---|
| Cột đơn NeurIPS | **1650 × 750** | Hầu hết các hội nghị ML/CV hạng A* (NeurIPS, ICML, ICLR) |
| Hai cột CVPR (dùng một cột) | **2062 × 850** | CVPR, ICCV, ECCV |
| Hai cột AAAI (dùng một cột) | **2025 × 825** | AAAI, IJCAI |
| Chiều rộng đầy đủ CHI (hiếm gặp) | **2400 × 1100** | CHI, UIST |

**Sử dụng khung vẽ mặc định của NeurIPS (1650 × 750)** — kích thước này có thể cắt (crop) gọn gàng để chuyển sang các định dạng khác.

### 5.1 Bố cục Hình học của Cột

- **5 cột**, mỗi cột **rộng 280 px**, cách nhau một khoảng **ngang 30 px** → 5×280 + 4×30 = 1520 px, chừa lại 65 px ở mỗi bên làm lề.
- **Thanh tiêu đề cột**: cao 40 px, nằm ở trên cùng của mỗi cột.
- **Các hộp thành phần**: rộng 240 px (thụt lề 20 px so với cột), chiều cao thay đổi từ 60–90 px. Xếp chồng lên nhau với **khoảng cách dọc là 20 px**.
- **Phạm vi dọc của cột**: lề trên 80 px → tổng cộng 580 px cho phần thân cột → lề dưới 90 px dành cho cung phản hồi và phần chú giải.

### 5.2 Thiết kế Hình học của Mũi tên

- **Các mũi tên luồng dữ liệu chính**: kết nối từ **cạnh phải** của cột $i$ đến **cạnh trái** của cột $i+1$, vẽ dưới dạng **đường thẳng ngang 2 px với một đầu mũi tên hình tam giác tô đặc**.
- **Mũi tên không được đè vào phần tô nền của cột** — bắt đầu chính xác từ viền phải của cột trước đó và kết thúc chính xác tại viền trái của cột tiếp theo. Đây là quy ước phổ biến nhất tại các hội nghị A*.
- **Cung phản hồi**: là một **đường cong Bezier** đơn giản với **hai điểm điều khiển (control points)**, vẽ với **độ dày viền 1.5 px, `dasharray: 6 4`**, màu sắc `#C0392B`. Các điểm điều khiển nên nằm bên dưới đường cơ sở của các cột khoảng ~120 px.

### 5.3 Đường phân chia quy trình phụ (Chỉ áp dụng cho Cột 2)

Bên trong cột Perception, vẽ một **đường đứt nét dọc 0.75 px** (`dasharray: 2 3`) tại vị trí x = (mép trái cột + 130 px), chia cột thành hai nửa: nửa bên trái "Body (RTMW3D)" và nửa bên phải "Face (OpenFace)". Đây là chỉ báo trực quan cho thấy "hai phương thức, cùng một cột" — không cần vẽ thêm cột thứ hai.

---

## 6. Quy trình Vẽ Toàn diện (End-to-End Drawing Workflow)

Đây là hướng dẫn từng bước chi tiết "cần nhấp vào các nút nào". Khuyến nghị về công cụ nằm ở phần §6.1; nếu bạn đã mở sẵn draw.io, có thể bỏ qua và chuyển sang phần §6.2.

### 6.1 Lựa chọn Công cụ — Chọn một trong số sau

| Công cụ | Ưu điểm | Nhược điểm | Khuyến nghị |
|---|---|---|---|
| **draw.io (diagrams.net)** | Miễn phí, không cần cài đặt, xuất được PDF/SVG, bạn đã có sẵn file nguồn `fig1_architecture.drawio` | Căn chỉnh thủ công, không hỗ trợ dựng công thức toán | **Nên sử dụng** nếu muốn lặp và thử nghiệm nhanh |
| **Figma** | Thiết kế mặc định đẹp mắt, dễ căn chỉnh, miễn phí cho tài khoản cá nhân, chia sẻ kiểu mẫu (styles) dễ dàng | Không hỗ trợ xuất LaTeX gốc trực tiếp; cần xuất ra PDF rồi `trích xuất` từ PDF | Nên dùng nếu bạn muốn hình ảnh có độ chau chuốt và thẩm mỹ cao nhất |
| **Inkscape** | Kiểm soát hoàn toàn, miễn phí, hỗ trợ xuất PDF/SVG gốc | Khó làm quen nhất, căn bắt điểm (snapping) thủ công | Bỏ qua trừ khi bạn đã sử dụng thành thạo từ trước |
| **TikZ (LaTeX)** | Nguồn mã duy nhất và thống nhất, đồng bộ phông chữ hoàn hảo với nội dung bài báo | Mất nhiều thời gian để chỉnh sửa, chỉ viết bằng mã code | **Chỉ dùng nếu người hướng dẫn yêu cầu** |
| **PowerPoint / Keynote** | Phổ biến, tạo hình vẽ đơn giản dễ dàng | Khó khăn khi căn chỉnh, không hỗ trợ xuất định dạng vector chuẩn | **Tránh sử dụng** đối với bài nộp hội nghị A* |

**Khuyến nghị**: bắt đầu vẽ trên **draw.io** để nhanh chóng định hình cấu trúc, sau đó tinh chỉnh thẩm mỹ trên **Figma** nếu có thời gian.

### 6.2 Các bước thực hiện trên draw.io (15 bước)

1. **Tạo biểu đồ mới** → File → New. Cấu hình kích thước khung vẽ thành 1650 × 750 px (Extras → Edit Diagram).
2. **View → Grid** → Chọn lưới 10 px, bật bắt điểm (snap to grid). Điều này giúp các thành phần có sự cân đối đồng đều.
3. **Vẽ 5 hình nền cho các cột** bằng các hình chữ nhật bo góc (`rounded=1;arcSize=4`):
   - Kích thước: 280 × 580, tọa độ (x, y) = (65, 80), (375, 80), (685, 80), (995, 80), (1305, 80).
   - Màu tô (Fill): `#F4F6F7` cho các cột 1, 3, 4, 5; `#D9E2F3` cho cột 2.
   - Viền (Stroke): `#566573` cho 1, 3, 4, 5; `#1F4E79` (độ dày 2 px) cho cột 2.
4. **Thêm tiêu đề cột** dưới dạng các nhãn chữ đè lên nền của từng cột tương ứng:
   - Nội dung chữ: "SENSING", "PERCEPTION", "ANALYSIS", "INTELLIGENCE", "INTERFACE".
   - Phông chữ: Arial Bold, 14 pt, màu `#1B2631` (hoặc màu `#FFFFFF` nếu bạn đặt tiêu đề *bên trong* một thanh màu ở trên cùng của cột — cả hai kiểu này đều đúng chuẩn hội nghị A*; chữ trắng trên nền màu là phong cách phổ biến hơn tại NeurIPS).
5. **Vẽ 13 hộp thành phần** (kích thước chi tiết ở mục §2):
   - Tất cả đều tô nền màu trắng, nét viền `#2C3E50` độ dày 1.25 px, bán kính bo góc 4.
   - Mỗi hộp có kích thước: 240 × 60 (đối với tiêu đề dòng đơn), hoặc 240 × 80 (đối với tiêu đề đi kèm nhãn phụ).
   - Căn giữa các hộp theo chiều ngang trong cột tương ứng.
6. **Bên trong cột Perception**, vẽ **đường phân chia đứt nét 0.75 px** tại x = mép trái cột + 140 px, chạy dọc theo chiều cao cột. Kiểu thiết kế: `strokeColor=#566573;dashed=1;dashPattern=2 3`.
7. **Thêm **Score (6-D) chip** vào cột Intelligence** dưới dạng một hình chữ nhật bo góc nhỏ hơn (160 × 28 px), đặt ở phía bên phải của hộp LLM Coach, tô nền `#FADBD8`, nét viền `#C0392B` (1 px).
8. **Vẽ 4 mũi tên luồng dữ liệu chính** kết nối giữa các cột:
   - Đi từ mép phải của cột $i$ (tọa độ x = i_left + 280, y = 370) → mép trái của cột $i+1$ (tọa độ x = (i+1)_left, y = 370).
   - Nét vẽ `#566573`, 2 px, có một đầu mũi tên hình tam giác, không uốn cong.
9. **Thêm nhãn cho các mũi tên** nằm phía trên mỗi mũi tên tại y = 350, phông 9 pt:
   - "RGB + audio", "3D KP · AU · text", "Angles + scores", "Coaching + alerts"
10. **Vẽ cung phản hồi (feedback arc)**:
    - Insert → Shape → **Bezier curve** (hoặc trong draw.io có thể chọn hình dạng "Bend", hoặc dùng mũi tên với cấu hình `curved=1`).
    - Điểm bắt đầu: mép dưới chính giữa của cột 5 (Interface), tọa độ (1445, 660).
    - Điểm kết thúc: mép dưới chính giữa của cột 4 (Intelligence), tọa độ (1135, 660).
    - Các điểm điều khiển: kéo phần võng của đường cong xuống y = 720 để tạo thành một cung nông kết nối 2 cột.
    - **Hoặc** đối với phiên bản mở rộng chuẩn hội nghị A*: kéo dài qua cả 5 cột — bắt đầu tại (1445, 660), kết thúc tại (1135, 660), các điểm điều khiển đặt tại (1445, 730) và (1135, 730). Cách này sẽ làm cho cung vẽ hiển thị rõ ràng *bên dưới* cả 5 cột.
    - Kiểu thiết kế: `strokeColor=#C0392B;strokeWidth=1.5;dashed=1;dashPattern=6 4;endArrow=classic`.
11. **Thêm nhãn phản hồi** ở điểm chính giữa của cung: "Pain / fatigue → adapt coaching" — 9 pt in nghiêng, màu `#C0392B`.
12. **Thêm một chú giải (legend) nhỏ** ở góc dưới cùng bên trái (hoặc góc trên cùng bên phải, tùy ý):
    - 4 dòng chữ với kích thước 8 pt:
      - `▬▬▬  Data flow`
      - `--- --  Safety feedback`
      - `▢  AI / learned component`
      - `▣  Rule-based component` (chỉ dùng nếu bạn muốn phân biệt rõ các thành phần — tùy chọn)
    - Vẽ bằng hộp bo góc kích thước 80 × 80 px, tô nền `#FFFFFF`, nét viền `#566573` (0.5 px), độ mờ (opacity) 95%.
13. **Kiểm tra độ chính xác**: mở bảng **Zoom**, chỉnh về 200%. Đứng ở góc nhìn của người phản biện để xem hình. Bạn có thể gọi tên 5 cột này trong vòng 2 giây không? Bạn có thể nhận ra 3 đóng góp khoa học chính từ hình vẽ hay không (3D pose, LLM coaching, safety)? Nếu không thể, thứ bậc trực quan của bạn đang bị sai.
14. **Lưu dưới dạng tệp `.drawio`** (để có thể chỉnh sửa sau này). Tên tệp: `paper/figures/fig1_architecture_v2.drawio`.
15. **Xuất ra tệp PDF**:
    - File → Export as → PDF.
    - **Crop** (Cắt sát viền): ✓
    - **Transparent background** (Nền trong suốt): ✗ (tắt tính năng này để giữ nền trắng phục vụ cho bản in)
    - **Selection only** (Chỉ vùng chọn): ✗
    - **Zoom**: 200% (định dạng PDF mặc định của draw.io thường quá nhỏ đối với tiêu chuẩn hội nghị A*)
    - Lưu với tên `paper/figures/fig1_architecture.pdf` (ghi đè lên phiên bản ATC cũ — hãy nhớ **sao lưu trước**: `cp paper/figures/fig1_architecture.pdf paper/figures/fig1_architecture_ieee_atc.pdf`).

### 6.3 Các bước thực hiện trên Figma (phương án thay thế)

Nếu bạn thấy giao diện của draw.io quá thô sơ, Figma sẽ cung cấp các tùy chọn mặc định sắc nét và chuyên nghiệp hơn:

1. Tạo tệp mới (New file) → Tạo khung vẽ (Frame) → Kích thước 1650 × 750.
2. Bật chức năng lưới căn chỉnh **Layout Grids** → Lưới 10 px, bật bắt điểm (snap on).
3. **Tạo một Thành phần (Component)** cho hình nền cột (kích thước 280 × 580, bo góc 4, tô màu trung tính). Nhân bản nó ra 5 lần. Đây là tính năng đắt giá nhất của Figma — bạn chỉ cần thay đổi màu nền một lần, tất cả 5 cột sẽ tự động cập nhật theo.
4. **Tạo một Thành phần (Component)** cho các hộp nội dung (kích thước 240 × 60, bo góc 4, tô nền trắng, viền 1.25 px). Sử dụng lại nó 13 lần.
5. Sử dụng phím tắt **Shift+kéo chuột** để nhân bản và căn chỉnh thẳng hàng nhanh chóng.
6. Xuất file: Share → Export → Chọn định dạng PDF, tắt các dấu căn lề (marks off), bleed 0.
7. Lưu nguồn `.fig` với tên `paper/figures/fig1_architecture.fig` để chỉnh sửa sau này.

---

## 7. Tích hợp mã LaTeX (LaTeX Integration)

Cập nhật file `paper/sections/methodology.tex` để chuyển sang bố cục một cột duy nhất theo đúng chuẩn của hội nghị A*.

### 7.1 Thay thế khối hình ảnh (Figure Block)

Trong tệp `paper/sections/methodology.tex`, khối mã hiện tại (dòng 18-23) là:

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/fig1_architecture.pdf}
    \caption{System architecture of \system{} showing the five-layer pipeline: Input, Perception, Analysis, Intelligence, and Output. The feedback loop enables real-time coaching adjustments based on detected pain and fatigue levels.}
    \label{fig:architecture}
\end{figure*}
```

Thay thế bằng mã phiên bản một cột chuẩn hội nghị A* (lưu ý dùng môi trường `figure` thay vì `figure*`, và vị trí đặt hình `[t]` để hình vẽ hiển thị ở phần trên cùng của cột báo):

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig1_architecture.pdf}
    \caption{Overview of the \system{} pipeline. RGB-D and audio are processed by RTMW3D-L (3D pose, 133 keypoints) and OpenFace~3.0 (8 AUs + 8 emotions); joint angles, smoothness, and compensation are extracted in the analysis layer; an LLM coach with RAG-grounded safety guardrails generates Vietnamese-language feedback; pain and fatigue detected at the interface close the feedback loop to adapt coaching in real time.}
    \label{fig:architecture}
\end{figure}
```

**Tại sao nội dung chú thích (caption) lại khác biệt**: Chú thích của hội nghị A* mở đầu trực tiếp bằng **đóng góp khoa học** ("Overview of the pipeline"), thay vì cách mô tả cấu trúc chung chung ("shows the five-layer pipeline"). Câu chú thích mới ở trên đã liệt kê đủ cả bốn điểm được **nhấn mạnh trực quan** trên hình vẽ: tư thế 3D pose (cột màu xanh dương), huấn luyện viên LLM coach, bộ lọc an toàn RAG safety, và cung phản hồi đau/mệt mỏi pain/fatigue.

### 7.2 Nếu Hội nghị A* hướng tới là NeurIPS (cột đơn, 5.5")

Bạn đã hoàn thành xong phần cấu hình. Tham số `\columnwidth` mặc định sẽ tự động tính toán để co giãn về kích thước 5.5".

### 7.3 Nếu Hội nghị A* hướng tới là CVPR/AAAI (hai cột, 6.75"/6.875")

Các hội nghị Thị giác máy tính hạng A* hầu như luôn đặt Hình 1 ở **trên cùng của cột 1 hoặc cột 2**, dưới dạng cột đơn. Tuy nhiên, nếu bài báo của bạn thiên nhiều về phương pháp phát triển và người phản biện kỳ vọng một sơ đồ rộng, hãy sử dụng:

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/fig1_architecture.pdf}
    \caption{...same caption as above...}
    \label{fig:architecture}
\end{figure*}
```

**Quan trọng**: đối với định dạng hình rộng, **hãy vẽ lại hình trên khung vẽ rộng** (2400 × 850, xem phần §5) — phiên bản 1650 × 750 trông sẽ bị co giãn không tự nhiên nếu bạn ép nó vừa khít 92% chiều rộng của trang văn bản 6.75". Các hộp thành phần lúc này cần phải có tỷ lệ nhỏ hơn một chút so với độ rộng của cột nền.

### 7.4 Cập nhật văn bản tham chiếu

Tìm kiếm chuỗi `Fig.~\ref{fig:architecture}` trong tệp `methodology.tex`. Nội dung tham chiếu ở dòng số 8 có thể giữ nguyên:

```latex
\system{} follows a five-column pipeline architecture as illustrated in Fig.~\ref{fig:architecture}.
```

(Đã chuyển từ "five-layer" → "five-column" để đồng bộ với nhãn cột trực quan mới "SENSING / PERCEPTION / ANALYSIS / INTELLIGENCE / INTERFACE". Đây tuy là một thay đổi nhỏ nhưng cực kỳ quan trọng về mặt sử dụng từ ngữ.)

### 7.5 Các tệp `.tex` khác chứa tham chiếu đến Hình 1

Chạy lệnh grep sau để tìm toàn bộ các tham chiếu liên quan và cập nhật chúng:

```bash
grep -rn "fig:architecture\|Figure 1\|Figure~1\|Fig\.~1\|Fig\. 1" paper/sections/
```

Mỗi vị trí gọi tham chiếu nên sử dụng cú pháp `Fig.~\ref{fig:architecture}` — dấu ngã (`~`) ngăn không cho xảy ra việc ngắt dòng giữa chữ "Fig." và chữ số phía sau, tuân thủ đúng chuẩn định dạng IEEE/NeurIPS.

---

## 8. Bảng kiểm tra chất lượng (Hãy in ra)

Trước khi gửi bài báo đi, hãy rà soát kỹ lưỡng qua danh sách kiểm tra sau:

**Thứ bậc trực quan (Visual hierarchy)**
- [ ] Tiêu đề của 5 cột phải là **điểm đầu tiên** mà mắt người đọc hướng tới (sử dụng kích thước chữ lớn nhất, màu sắc mạnh mẽ nhất).
- [ ] Đóng góp về pose 3D (Cột 2) là **cột duy nhất được tô màu nền**. Người phản biện phải có thể tìm thấy nó ngay lập tức trong vòng 1 giây.
- [ ] Cung phản hồi (feedback arc) là **thành phần màu đỏ duy nhất** xuất hiện trong hình. Nếu dùng quá nhiều màu đỏ ở các vị trí khác, thông điệp về tính an toàn sẽ bị loãng đi.
- [ ] Không có bất kỳ hộp thành phần nào rộng hơn chiều rộng cột nền của nó. (Hãy sử dụng mép viền cột làm rào chắn giới hạn.)
- [ ] Chữ không được tự động xuống dòng lỗi trong hộp — toàn bộ nhãn phải hiển thị vừa vặn trên một dòng, hoặc hộp phải đủ cao để chứa 2 dòng chữ.

**Kiểu chữ (Typography)**
- [ ] Toàn bộ chữ trong hình phải thuộc **một phông chữ duy nhất** (Helvetica hoặc Arial).
- [ ] Toàn bộ chữ đều đạt kích thước **≥ 8 pt ở kích thước in cuối cùng**. Chữ nhỏ nhất trên sơ đồ là phần chú giải (8 pt) và nhãn của các mũi tên (9 pt) — hãy kiểm tra lại bằng cách phóng to lên 200% trong trình xem PDF.
- [ ] Không in nghiêng đối với tiêu đề cột hoặc tiêu đề hộp (chữ in nghiêng chỉ dành riêng cho nhãn của cung phản hồi).

**Màu sắc (Color)**
- [ ] Sử dụng tối đa 2 màu nhấn (xanh dương + đỏ) kết hợp với dải màu xám trung tính. **Hãy loại bỏ** toàn bộ các màu xanh lá, vàng, hay tím.
- [ ] Chuyển đổi tệp PDF sang dạng thang màu xám (grayscale) để kiểm tra (Preview → Export as PNG → Filters → B&W). Cả 5 cột phải hiển thị rõ nét và phân biệt được với nhau. Nếu cột "Perception" bị lẫn hoàn toàn vào cột "Analysis", hãy làm cho nền của nó tối hơn hoặc tăng độ đậm viền của cột đó lên.
- [ ] Chạy kiểm thử hình vẽ thông qua công cụ mô phỏng mù màu (Coblis, Sim Daltonism). Cặp màu xanh dương/đỏ thông thường là an toàn; bạn chỉ cần kiểm tra lại để chắc chắn.

**Các mũi tên (Arrows)**
- [ ] Cả 4 mũi tên chính đều phải **nằm ngang, là nét đơn, độ dày viền đạt 2 px**.
- [ ] Cung phản hồi phải là **một đường cong nét đứt liên tục**, không ghép nối từ nhiều đoạn rời rạc.
- [ ] Không vẽ mũi tên đè cắt ngang qua bất cứ một hộp nào. Nếu mũi tên buộc phải đi qua cột, hãy căn tuyến cho nó đi **xuyên qua khoảng trống giữa** các hộp (khoảng cách dọc giữa các hộp).

**Tính chính xác về mặt kỹ thuật (Engineering correctness)** (đối chiếu với `main_v3.py` và `methodology.tex`)
- [ ] RTMW3D-L → 133 điểm mốc (keypoints) ✓
- [ ] OpenFace 3.0 → 8 AU + 8 trạng thái cảm xúc (emotions) ✓
- [ ] Whisper được đặt ở cột Perception, chứ không phải ở cột Intelligence (nhiệm vụ chuyển đổi âm thanh sang chữ viết thuộc về nhận thức - perception, không phải là ra quyết định) ✓
- [ ] Edge-TTS được đặt ở cột Intelligence (vì *huấn luyện viên* sẽ ra quyết định thời điểm phát ngôn, giao diện chỉ hiển thị và phát ra tiếng) ✓
- [ ] Chip tính điểm 6 chiều (6-D score chip) được đặt trong cột Intelligence chứ không nằm ở cột Analysis (đây là *đầu ra* của quá trình phân tích và được sử dụng trực tiếp bởi huấn luyện viên) ✓
- [ ] Cung phản hồi phải bắt đầu từ cột Interface (nơi phát hiện ra cơn đau - pain *detected*) và kết thúc ở cột Intelligence (nơi huấn luyện viên *quyết định hành động ứng phó*) ✓

**Sẵn sàng nộp bài (Submission-ready)**
- [ ] Tệp PDF phải ở định dạng **vector** (chữ trên hình có thể bôi đen và chọn được trong trình xem PDF — hãy kiểm thử bằng cách chọn chữ "RTMW3D-L" bằng công cụ Text).
- [ ] Kích thước tệp tin nhỏ hơn < 1 MB (kích thước hình vẽ chuẩn hội nghị A* thường dao động từ 100–500 KB).
- [ ] Tên tệp tin trùng khớp chính xác với phần tham chiếu trong file `.tex`: `paper/figures/fig1_architecture.pdf`.
- [ ] Đã sao lưu phiên bản ATC cũ thành tệp tin `paper/figures/fig1_architecture_ieee_atc.pdf`.
- [ ] Bản nguồn gốc có thể chỉnh sửa đã được sao lưu thành tệp `paper/figures/fig1_architecture_v2.drawio` (hoặc `.fig`).
- [ ] Chú thích trong `methodology.tex` khớp với các điểm được nhấn mạnh trực quan trên sơ đồ mới.
- [ ] Tất cả các tham chiếu chéo `Fig.~\ref{fig:architecture}` đều được giải quyết chuẩn xác (biên dịch LaTeX và kiểm tra xem có xuất hiện lỗi ký hiệu `??` hay không).

---

## 9. Tham chiếu nhanh: Các chuỗi định dạng (Style Strings - Sao chép trực tiếp vào XML của draw.io)

```
Nền cột (trung tính - neutral):
rounded=1;whiteSpace=wrap;html=1;fillColor=#F4F6F7;strokeColor=#566573;strokeWidth=1.25;arcSize=4;

Nền cột (Cột Perception — màu nhấn xanh dương):
rounded=1;whiteSpace=wrap;html=1;fillColor=#D9E2F3;strokeColor=#1F4E79;strokeWidth=2;arcSize=4;

Hộp thành phần (Component box):
rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2C3E50;strokeWidth=1.25;arcSize=5;

Chip tính điểm (màu nhấn đỏ - Score chip):
rounded=1;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#C0392B;strokeWidth=1;arcSize=8;

Mũi tên luồng chính (Main flow arrow):
endArrow=classic;html=1;strokeColor=#566573;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;

Cung phản hồi (nét đứt màu đỏ - Feedback arc):
endArrow=classic;html=1;strokeColor=#C0392B;strokeWidth=1.5;dashed=1;dashPattern=6 4;curved=1;

Đường phân chia luồng phụ (nét đứt màu xám - Sub-pipeline divider):
endArrow=none;html=1;strokeColor=#566573;strokeWidth=0.75;dashed=1;dashPattern=2 3;
```

---

## 10. Cách xử lý đối với các tệp tin cũ

Không xóa các tệp tin này đi — thỉnh thoảng những người phản biện bài hội nghị A* sẽ xem xét tài liệu bổ trợ (supplementary material), ngoài ra bản lưu trữ tài liệu đã nộp cho ATC trước đây cũng cần được giữ lại. Hãy di chuyển chúng sang vị trí lưu trữ riêng:

```bash
cd /home/haipd/ADAPT-Rehab
mkdir -p paper/figures/legacy_ieee_atc/
mv paper/figures/fig1_architecture.pdf      paper/figures/legacy_ieee_atc/fig1_architecture.pdf
mv paper/figures/fig1_architecture.drawio   paper/figures/legacy_ieee_atc/fig1_architecture.drawio
mv paper/figures/fig1_architecture.mmd       paper/figures/legacy_ieee_atc/fig1_architecture.mmd
# Keep fig1_drawio_guide.md in place — it's a useful A/B comparison
```

Now `paper/figures/fig1_architecture.pdf` is free for the new A* version.

---

## 11. Sơ lược Tóm tắt một Trang (Hãy in ra và dán trên bàn làm việc của bạn)

| Thành phần / Yêu cầu | Giá trị chuẩn hội nghị A* |
|---|---|
| **Khung vẽ (Canvas)** | 1650 × 750 px (NeurIPS), 2400 × 850 (hình rộng) |
| **Các cột (Columns)** | 5 cột, cùng chiều rộng (280 px), khoảng cách ngang 30 px |
| **Số hộp trên mỗi cột** | 2–3 hộp (Input 2, Perception 3, Analysis 3, Intelligence 3+1 chip phụ, Interface 2) |
| **Tổng số hộp** | 13 hộp + 1 chip phụ |
| **Màu sắc nhấn** | Xanh dương `#1F4E79` (Perception) + Đỏ `#C0392B` (Phản hồi an toàn) |
| **Màu trung tính** | `#F4F6F7` (tô nền cột), `#FFFFFF` (nền hộp), `#2C3E50` (viền hộp), `#1B2631` (chữ chính) |
| **Phông chữ (Font)** | Helvetica hoặc Arial, dùng đồng bộ một phông chữ |
| **Kích thước tiêu đề cột** | 14 pt bold (in đậm) |
| **Tiêu đề hộp** | 12 pt bold |
| **Nhãn phụ (Sub-label)** | 9.5 pt regular (chữ thường) |
| **Nhãn mũi tên** | 9 pt regular |
| **Kích thước chữ nhỏ nhất** | 8 pt (chỉ áp dụng đối với phần chú giải) |
| **Phong cách mũi tên** | 2 px nét liền màu `#566573` cho luồng chính, 1.5 px nét đứt màu `#C0392B` cho cung phản hồi |
| **Nghiêm cấm** | sử dụng gradient, đổ bóng (shadow), sử dụng quá 2 màu nhấn, in nghiêng tiêu đề |
| **Môi trường LaTeX** | `figure[t]` (hình cột đơn) hoặc `figure*[t]` (hình rộng) |
| **Mở đầu của chú thích** | "Overview of the \system{} pipeline" (nêu bật đóng góp chính trước tiên) |
| **Chú thích (Caption)** | nhắc đến tư thế 3D pose, LLM coach, độ an toàn RAG safety, vòng lặp phản hồi đau/mệt mỏi |
| **Đầu ra tệp tin** | Định dạng PDF vector, nền trắng, dung lượng dưới < 1 MB |
| **Sao lưu nguồn** | Định dạng `.drawio` (hoặc `.fig`) lưu trong thư mục `paper/figures/` |

---

## 12. Sau khi Hoàn thành — Các lệnh Kiểm thử

Chạy các dòng lệnh sau từ thư mục gốc của dự án sau khi bạn đã xuất ra tệp PDF mới:

```bash
# 1. Kiểm tra định dạng PDF là vector (chữ có thể bôi đen để chọn được, không bị rasterized/hóa ảnh)
pdftotext paper/figures/fig1_architecture.pdf - | head -20
# Kết quả kỳ vọng: sẽ thấy các chuỗi "RTMW3D-L", "PERCEPTION", v.v. dưới dạng văn bản thô

# 2. Kiểm tra dung lượng tệp tin dưới 1 MB
ls -lh paper/figures/fig1_architecture.pdf
# Kết quả kỳ vọng: khoảng ~100-500 KB

# 3. Kiểm tra các tham chiếu LaTeX hoạt động bình thường
cd paper && pdflatex -draftmode main.tex 2>&1 | grep -i "fig:architecture\|undefined"
# Kết quả kỳ vọng: không có cảnh báo lỗi "undefined reference"

# 4. Kiểm tra kích thước hình ảnh khớp chính xác với độ rộng của cột báo
pdfinfo paper/figures/fig1_architecture.pdf | grep -i "page size"
# Kết quả kỳ vọng: dạng tương tự như "Page size: 595 x 270 pts" (đối với định dạng cột đơn NeurIPS ở 300 dpi)
# Nếu kích thước hiển thị 595x842 (khổ A4), tức là bạn đã quên bước cắt sát viền (crop)
```

Nếu cả bốn bước kiểm tra trên đều vượt qua thành công, bạn đã sẵn sàng để gửi bài báo. Chúc bạn gặp nhiều may mắn.
