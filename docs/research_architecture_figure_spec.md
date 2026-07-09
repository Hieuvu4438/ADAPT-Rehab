# Đặc Tả Kiến Trúc Sơ Đồ Hệ Thống Cho Research Paper (Hội Nghị A*)

Tài liệu này cung cấp bản đặc tả chi tiết toàn diện (Specification) đã được tinh gọn để vẽ sơ đồ kiến trúc hệ thống (System Architecture Figure) của dự án **ADAPT-Rehab** chuẩn hội nghị khoa học xếp hạng A* (CVPR, AAAI, NeurIPS, CHI). Sơ đồ này tập trung vào các cấu phần cốt lõi: xử lý thị giác (Vision), phân tích động học (Kinematics), đánh giá lâm sàng và huấn luyện viên LLM.

---

## 1. Hệ Quy Chuẩn Màu Sắc (Color Palette Specification)

Để đảm bảo hình vẽ hiển thị xuất sắc cả trên màn hình màu và khi in đen trắng (B&W/Photocopy), hệ thống sử dụng quy chuẩn **2 Màu Nhấn (Two-Accent System)** kết hợp dải màu trung tính:

| Nhóm Màu | Tên Màu | Mã Hex | Vai Trò & Cách Dùng |
| :--- | :--- | :--- | :--- |
| **Primary Accent** | Xanh Dương Đậm (Dark Blue) | `#1F4E79` | Tiêu đề cột Perception, viền các khối nhận thức vật lý (3D Pose). |
| **Primary Fill** | Xanh Dương Nhạt (Light Blue) | `#D9E2F3` | Tô nền cột Perception và các khối con thuộc luồng nhận thức. |
| **Secondary Accent** | Đỏ Ấm (Warm Red) | `#C0392B` | Viền các cấu phần an toàn (Safety Guardrails) và cung phản hồi Đau/Mỏi. |
| **Secondary Fill** | Đỏ Nhạt (Light Red) | `#FADBD8` | Tô nền khối Safety Guardrails và chip chấm điểm. |
| **Neutral Border** | Xanh Đen (Deep Slate Blue) | `#2C3E50` | Đường viền của các khối chức năng thông thường (độ dày 1.25 px). |
| **Neutral Fill (Body)** | Xám Nhạt (Very Light Gray)| `#F4F6F7` | Nền các cột Sensing, Analysis, Intelligence, Interface. |
| **Neutral Box Fill** | Trắng Tuyệt Đối (Pure White) | `#FFFFFF` | Nền của các hộp chức năng bên trong cột. |
| **Main Text** | Đen Gần Tuyệt Đối (Near Black) | `#1B2631` | Tiêu đề lớn, tên khối chính (độ tương phản cao). |
| **Secondary Text** | Xám Đậm (Medium Gray) | `#566573` | Nhãn phụ, thông số kỹ thuật bên trong khối và nhãn mũi tên. |

---

## 2. Bảng Phân Rã Các Module & Khối Chức Năng (Module & Block Breakdown)

Sơ đồ được tổ chức thành **5 cột giai đoạn ngang** (Backbone) và **3 bảng phóng to (Call-out Panels)** ở biên.

### 2.1 Các Cột Trục Chính (Backbone Columns)

| ID Khối | Tên Khối Chính (12pt Bold, `#1B2631`) | Nội Dung Phụ (9.5pt Regular, `#566573`) | Nền Khối (Fill) | Viền Khối (Border) | Vai Trò & Điểm Nhấn |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **COL 1** | **SENSING (Tầng Vào)** | *Nền cột: `#F4F6F7`* | | Viền: `#566573` | |
| B1.1 | **RGB Camera / Webcam** | 30 FPS, 1080p Video Input | `#FFFFFF` | `#2C3E50` | Nguồn thu nhận hình ảnh thô |
| **COL 2** | **PERCEPTION (Nhận Thức)** | *Nền cột: `#D9E2F3`* | | Viền: `#1F4E79` | **Điểm nhấn chính (Primary)** |
| B2.1 | **RTMW3D-L** | Whole-body 3D Pose, 133 KP | `#FFFFFF` | `#1F4E79` (2px) | Ước lượng khung xương 3D |
| B2.2 | **OpenFace 3.0** | 8 Action Units, Emotion, Gaze | `#FFFFFF` | `#1F4E79` (2px) | Nhận diện biểu cảm nét mặt |
| **COL 3** | **ANALYSIS (Phân Tích)** | *Nền cột: `#F4F6F7`* | | Viền: `#566573` | |
| B3.1 | **Quaternion Kinematics** | 8 Bilateral Joints, ISB standard | `#FFFFFF` | `#2C3E50` | Tính toán góc khớp động học |
| B3.2 | **Constrained DTW** | Sakoe-Chiba Band Alignment | `#FFFFFF` | `#2C3E50` | Đối sánh nhịp điệu chuyển động |
| B3.3 | **Compensation & Fatigue**| Joint Jitter & Body Drift | `#FFFFFF` | `#2C3E50` | Phát hiện mỏi cơ & ăn gian |
| **COL 4** | **INTELLIGENCE (Trí Tuệ)**| *Nền cột: `#F4F6F7`* | | Viền: `#566573` | |
| B4.1 | **LLM Coach** | GPT-4o / Claude Agent | `#FFFFFF` | `#2C3E50` | Bộ não lập luận phản hồi |
| B4.2 | **Safety Guardrails** | Contraindication Interceptor | `#FADBD8` | `#C0392B` (2px) | **Điểm nhấn phụ (Secondary)** |
| B4.3 | **Edge-TTS** | vi-VN Voice Synthesizer | `#FFFFFF` | `#2C3E50` | Tổng hợp giọng nói chỉ dẫn |
| **COL 5** | **INTERFACE (Đầu Ra)** | *Nền cột: `#F4F6F7`* | | Viền: `#566573` | |
| B5.1 | **Visual Overlay** | 3D Skeleton + Dynamic ROM Arcs| `#FFFFFF` | `#2C3E50` | Trực quan hóa góc khớp |
| B5.2 | **Voice Feedback** | Natural Audio Instructions | `#FFFFFF` | `#2C3E50` | Giọng nói chỉ dẫn phát ra loa |
| B5.3 (chip)| **Real-time HUD** | Rep Counter & Session Scores | `#FFFFFF` | `#2C3E50` | Bảng thông số hiển thị |

---

### 2.2 Các Bảng Con Phóng To (Detailed Call-out Panels)

| ID Panel | Tên Panel (14pt Bold) | Thuật Toán / Cấu Phần Bên Trong (10pt Regular) | Nền Panel | Viền Panel | Khối Gốc Được Phóng To |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PANEL A**| **Detailed Face Parser** | 1. `EAR Calculator` (Eyelid aspect ratio)<br/>2. `PSPI Pain`: $\text{AU4} + 2\text{AU6} + \text{AU9} + 2\text{AU43}$<br/>3. `PERCLOS`: $\%$ frames eye closure $\ge 80\%$ | `#FFFFFF` | `#1F4E79` (2px, nét đứt) | B2.2 (OpenFace 3.0) |
| **PANEL B**| **Kinematic & Smoothness**| 1. `Butterworth Filter` (Order 4, Cutoff 6Hz)<br/>2. `SPARC Smoothness`: Arc length of Fourier spectrum<br/>3. `Melax Quaternion`: Vector rotation w/o gimbal lock | `#FFFFFF` | `#2C3E50` (1.5px, nét đứt) | B3.1 & B3.3 |
| **PANEL C**| **Safety Guardrail Loop** | 1. `Context Serializer` (JSON builder)<br/>2. `Safety Guardrail`: Threshold monitor<br/>3. `Emergency Interceptor`: Hijacks output on high Pain/Fatigue | `#FADBD8` | `#C0392B` (2px, nét đứt) | B4.2 (Safety Guardrails) |

---

## 3. Bảng Ánh Xạ Kết Nối (Connection & Data Flow Map)

Dưới đây là ma trận các mũi tên kết nối dòng chảy dữ liệu giữa các khối sau khi tinh gọn:

| ID Kết Nối | Từ Khối (Source) | Đến Khối (Target) | Loại Đường Vẽ | Nhãn Mũi Tên (9pt Text, `#1B2631`) | Giải Thích Luồng Dữ Liệu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1.1** | B1.1 (Camera) | B2.1 (RTMW3D) | Rắn, Đơn | `RGB Frame` | Gửi ảnh thô sang ước lượng pose |
| **L1.2** | B1.1 (Camera) | B2.2 (OpenFace) | Rắn, Đơn | `RGB Frame` | Gửi ảnh thô sang phân tích mặt |
| **L2.1** | B2.1 (RTMW3D) | B3.1 (Kinematics) | Rắn, Đơn | `3D Keypoints` | Tọa độ khớp truyền sang bộ tính góc |
| **L2.2** | B2.1 (RTMW3D) | B3.3 (Comp) | Rắn, Đơn | `3D Keypoints` | Tọa độ vai/thân sang bộ phát hiện ăn gian |
| **L2.3** | B2.2 (OpenFace) | B3.3 (Fatigue) | Rắn, Đơn | `AU intensities` | Gửi dữ liệu cơ mặt để đo mệt mỏi |
| **L3.1** | B3.1 (Kinematics) | B3.2 (DTW) | Rắn, Đơn | `Filtered Angles` | Gửi chuỗi góc khớp sang so khớp DTW |
| **L3.2** | B3.1 (Kinematics) | B3.3 (Comp) | Rắn, Đơn | `Angular Velocity` | Vận tốc góc phục vụ SPARC/Jerk |
| **L3.3** | B3.2 (DTW) | B4.1 (LLM Coach) | Rắn, Đơn | `Similarity Score`| Điểm nhịp điệu gửi về bộ xử lý LLM |
| **L3.4** | B3.3 (Comp/Fat) | B4.2 (Safety) | Rắn, Đơn | `Pain/Fatigue/Comp`| Các biến số chẩn đoán gửi về Safety |
| **L4.1** | B4.1 (LLM Coach) | B4.2 (Safety) | Rắn, Đơn | `Raw Text Output` | Đưa câu thoại huấn luyện viên sang bộ lọc |
| **L4.2** | B4.2 (Safety) | B4.3 (TTS) | Rắn, Đơn | `Safe Text` | Gửi câu thoại an toàn sang Edge-TTS |
| **L4.3** | B4.3 (TTS) | B5.2 (Voice) | Rắn, Đơn | `Audio stream` | Phát âm thanh giọng đọc ra loa |
| **L4.4** | B4.2 (Safety) | B5.1 (Visual) | Rắn, Đơn | `Control signals` | Điều khiển hiển thị màu sắc trên màn hình |
| **L_FB** | B5.3 (HUD/Scores) | B4.1 (LLM Coach) | Đứt, Uốn Cong | *Pain/Fatigue Feedback Loop* | **Cung phản hồi ngược thời gian thực** (màu đỏ `#C0392B`) chạy dưới đáy sơ đồ để LLM thay đổi giáo án tập |

---

## 4. Hướng Dẫn Vẽ Call-out Phóng To (Detailed View Connection)

Để liên kết giữa các cấu phần trên trục chính với các **Bảng Con Phóng To (Panel A, B, C)** mà không làm rối sơ đồ:

```
  Cột Trục Chính (Backbone)                    Bảng Phóng To Bên Ngoài (Panel)
 ┌──────────────────────┐                     ┌───────────────────────────────────┐
 │ B2.2: OpenFace 3.0   │ ── ── ── ── ── ── ┐ │ PANEL A: DETAILED FACE PARSER     │
 └──────────┬───────────┘ (Đường nét đứt    │ │ - EAR Calculator -> AU43          │
            │              kéo góc 1px)     │ │ - PSPI = AU4+2AU6+AU9+2AU43       │
            ▼                               └▶│ - PERCLOS calculation             │
     (Luồng chính)                            └───────────────────────────────────┘
```

1. **Kết nối Panel A**: Kéo đường nét đứt mảnh màu xanh dương nhạt từ khối `B2.2` sang `PANEL A`. Đặt Panel A ở vùng trống phía trên cột Perception.
2. **Kết nối Panel B**: Kéo đường nét đứt mảnh màu xám từ khối `B3.1` và `B3.3` sang `PANEL B`. Đặt Panel B ở vùng trống dưới cột Analysis.
3. **Kết nối Panel C**: Kéo đường nét đứt mảnh màu đỏ nhạt từ khối `B4.2` sang `PANEL C`. Đặt Panel C ở vùng trống dưới cột Intelligence.

---

## 5. Quy Tắc Định Dạng Kiểu Chữ (Typography Rules)

*   **Font Family**: Sử dụng một phông chữ Sans-serif duy nhất cho toàn bộ sơ đồ (khuyến nghị **Helvetica** hoặc **Arial**).
*   **Tiêu đề lớn của các cột**: `14 pt`, **Bold**, màu `#1B2631`.
*   **Tên khối chính**: `12 pt`, **Bold**, màu `#1B2631`.
*   **Nhãn phụ kỹ thuật**: `9.5 pt`, Regular, màu `#566573`.
*   **Nhãn trên các mũi tên**: `9 pt`, Regular, màu `#1B2631` hoặc `#566573`.
*   **Chú giải (Legend)**: `8 pt`, Regular, đặt gọn gàng ở góc dưới bên trái của khung vẽ.
