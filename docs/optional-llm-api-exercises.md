# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, câu trả lời mang tính tất định cao nhất, câu từ ngắn gọn, tập trung vào các sự thật phổ biến và uy tín nhất (như xuất khẩu hạt điều/cà phê Robusta). Khi tăng lên 0.5–1.0, từ vựng và chủ đề trở nên phong phú, sinh động hơn. Đến mức 1.5, mô hình trở nên quá ngẫu hứng, hành văn bay bổng nhưng bắt đầu xuất hiện cấu trúc câu lỏng lẻo và nguy cơ sai lệch thông tin thực tế.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng **0.0 đến 0.2**. Chatbot hỗ trợ khách hàng đòi hỏi sự chính xác tuyệt đối, nhất quán giữa các phiên hỏi đáp và bám sát tài liệu chính sách của doanh nghiệp; nhiệt độ thấp giúp triệt tiêu hiện tượng ảo giác (hallucination) về giá cả, chính sách đổi trả hay thông tin kỹ thuật.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> `gpt-6.1-terra` đắt hơn `gpt-6-luna` đúng **10 lần** cho toàn bộ workload này. Lý do: Cả đơn giá input ($1.00 so với $0.10) lẫn đơn giá output ($5.00 so với $0.50) của Terra đều gấp đúng 10 lần Luna. Với tổng khối lượng 10.000 × 3 × 350 = 10,5 triệu token/ngày, bất kể tỷ lệ phân bổ input/output là bao nhiêu, tổng chi phí của Terra luôn gấp 10 lần Luna (ví dụ: nếu chia đều 50/50 input/output thì Luna tốn ~$31.50/tháng còn Terra tốn ~$315.00/tháng).

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> - **Xứng đáng dùng `gpt-6.1-terra`:** Tác vụ phân tích hợp đồng pháp lý, chuẩn đoán dữ liệu y khoa hoặc tư vấn chiến lược đầu tư phức tạp — nơi một sai sót nhỏ do suy luận nông có thể gây thiệt hại tài chính hoặc pháp lý lớn hơn nhiều so với chi phí token.
> - **Nên dùng `gpt-6-luna`:** Tác vụ phân loại spam/ham, trích xuất thực thể tên riêng (NER), gắn thẻ phân loại ticket hỗ trợ hoặc gợi ý câu trả lời nhanh — nơi ngữ cảnh rõ ràng, yêu cầu độ trễ thấp và volume cực lớn hàng triệu request mỗi ngày.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện hội thoại tương tác trực tiếp với con người (như chatbot tư vấn, trợ lý viết mã), nơi việc hiển thị từng token ngay lập tức giúp giảm thời gian chờ phản hồi đầu tiên (Time to First Token - TTFT) từ vài giây xuống mili-giây, mang lại cảm giác phản hồi tức thì và không làm người dùng sốt ruột. Ngược lại, non-streaming phù hợp hơn với các tác vụ xử lý ngầm (background batch jobs), các luồng ETL trích xuất dữ liệu, hoặc khi gọi API yêu cầu định dạng JSON có cấu trúc (structured JSON output) mà hệ thống cần nhận trọn vẹn toàn bộ payload hợp lệ trước khi tiến hành parse và đưa vào cơ sở dữ liệu.

## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
