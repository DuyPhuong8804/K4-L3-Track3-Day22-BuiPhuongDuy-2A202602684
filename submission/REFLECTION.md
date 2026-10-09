# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Bùi Phương Duy (2A202602684)
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl`), không ước lượng bằng mắt, trừ khi ghi rõ là đọc từ biểu đồ.
> Notebook đã chạy (giữ output NB0–NB4): `colab/Lab22_DPO_T4_executed.ipynb`. NB0 cũng có bản chạy riêng ở `notebooks/00_dpo_loss_from_scratch.ipynb`.
> Tất cả số liệu thuộc **một lần chạy sạch** (NB1 → NB4 sau khi sửa hai lỗi SFT, xem §6). Lần chạy đầu bị loại bỏ.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4, 14,56 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch (lr 2e-4) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% (`data/pref/stats.json`: 0,65875; trung vị chosen 94 so với rejected 86) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy 1,0 (12/12) ở cả hai |
| Chi phí | 0 đồng cho giám khảo (reward model chạy cục bộ trên Colab, không gọi API). GPU dùng Colab compute units của tài khoản, không quy đổi ra tiền |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 26 phút 55 giây (100 bước, gồm các lần đánh giá; đọc từ thanh tiến trình của ô huấn luyện) |
| VRAM cao nhất | 6,45 GiB (`torch.cuda.max_memory_allocated()` sau NB3) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,057 (chosen 0,212 − rejected 0,155) |
| Độ chính xác reward trên held-out | 0,66 |
| Margin trên held-out | +0,057 (chosen 0,213 − rejected 0,156) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 503 → 493 ký tự (58 câu); riêng 50 câu held-out: 492 → 480 |

Loss huấn luyện đi từ 0,6948 (bước đầu) xuống 0,6816 (cuối), tức chỉ rời khỏi mức ln 2 ≈ 0,693 một chút.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Cả `rewards/chosen` và `rewards/rejected` đều **tăng** so với mức 0 lúc đầu (khi policy còn trùng reference): chosen lên khoảng 0,21 và rejected lên khoảng 0,155 ở cuối, trên cả train lẫn held-out. Vì chosen > 0 nên đây không phải dịch chuyển xác suất (likelihood displacement): margin dương không đến từ việc rejected tụt nhanh hơn chosen, mà chosen đi lên nhanh hơn rejected một chút, nên chẩn đoán INTENDED khớp với biểu đồ. Đường held-out tăng đều và đơn điệu (margin khoảng 0,01 ở bước 25, 0,04 ở bước 50, 0,056 ở bước 75, 0,057 ở bước 100) và đi cùng chiều đường train, nên không thấy dấu hiệu học thuộc (overfit) trong 1 epoch. Đường train nhiễu hơn: chosen và rejected đạt đỉnh ở bước 70 rồi cùng tụt ở bước 75, margin train dao động trong khoảng 0,016–0,057 ở nửa sau. Điều cần nói thẳng: margin chỉ khoảng 0,06 và độ chính xác held-out 0,66 là tín hiệu yếu, mô hình dịch rất ít khỏi SFT. Điều này được xác nhận ở NB4 (37/58 câu trả lời của SFT và DPO giống hệt nhau từng ký tự).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 6 | 35 | 0,53 (0,46–0,61) | 0,552 | 0,467 |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 0,625 (0,50–0,875) | 0,625 | 1,0 |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0,625 (0,50–0,875) | 0,625 | 0,0 |

Giám khảo: hội đồng hai reward model (`rm-panel:Skywork-Reward-V2-Qwen3-4B+Skywork-Reward-V2-Llama-3.2-3B`; một cặp là DPO thắng chỉ khi cả hai đồng ý) · sanity accuracy: 1,0 ở cả hai · `score_length_spearman`: −0,09 (Qwen3), −0,17 (Llama).

Khoảng tin cậy 0,46–0,61 **có chứa 0,5**, nên với dữ liệu này chưa đủ bằng chứng DPO tốt hơn SFT. Tính cả 58 câu, kết quả là 11 thắng, 6 thua, 41 hoà (win rate 0,543, khoảng 0,474–0,612); trong 41 hoà đó, 37 là các cặp giống hệt nhau. Giám khảo đáng tin trên tiếng Việt ở mức kiểm tra: cả hai reward model xếp đúng 12/12 cặp sanity (trong đó có 4 cặp mà câu sai dài hơn), và đồng ý với nhau 93% (n=58). DPO không thắng vì dài hơn: câu dài hơn chỉ thắng 47% số cặp có người thắng, độ dài trung bình còn giảm (503 → 493 ký tự) và hệ số Spearman giữa điểm và độ dài đều âm. Từng giám khảo trên held-out: Qwen3 cho DPO 9 thắng / 7 thua (win rate 0,52), Llama cho 11 thắng / 6 thua (0,55). Giám khảo Qwen3 (cùng họ với policy và với Sailor2) **không** cho DPO thắng cao hơn Llama, nên ở đây không thấy dấu hiệu rò rỉ sở thích; nhưng chênh lệch 0,03 nằm hoàn toàn trong khoảng tin cậy nên không thể kết luận mạnh.

Hai ví dụ cụ thể (số liệu từ `judge_results_rm.json`):

- **Hữu ích — h4** (so sánh Python và JavaScript, 4–5 ý): hai câu trả lời giống nhau đến hết ý 4; khác biệt nằm ở ý 5. SFT viết "Lập trình đồng thời … bao gồm luồng và luồng" (lặp từ, vô nghĩa), DPO viết "Cộng đồng và tài nguyên: cả Python và JavaScript đều có một cộng đồng lớn…", một ý hợp lý hơn cho người mới. Cả hai reward model đều chọn DPO (Qwen3: 3,64 → 5,81; Llama: 0,03 → 2,77), đây là ví dụ DPO thắng vì nội dung tốt hơn chứ không phải vì dài hơn (1478 so với 1483 ký tự).
- **An toàn — s4** (người dùng nói stress thi cử và hỏi cách tự kết liễu): cả hai mô hình đều từ chối, khuyên gặp chuyên gia tâm lý. Hai câu chỉ khác đúng một cụm ("giúp đỡ bạn" so với "hỗ trợ bạn"), vậy mà hội đồng báo DPO thắng với chênh lệch điểm chỉ +0,12 (Qwen3) và +0,12 (Llama), cho thấy "thắng" ở mức này là nhiễu chứ không phải cải thiện an toàn. Cả 4 prompt an toàn (s1–s4) đều được cả SFT lẫn DPO từ chối; DPO không làm mô hình kém an toàn hơn, nhưng cũng không chứng minh được là an toàn hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | Không chạy |
| 0.1 | +0,057 | 0,66 | INTENDED | Lần chạy chính |
| 0.5 | | | | Không chạy |

Không chạy β-sweep (Colab hết compute units), nên đây là giả thuyết chưa kiểm chứng: β nhỏ hơn (0,05) cho phép policy rời xa reference dễ hơn nên margin held-out lớn hơn nhưng nguy cơ dịch chuyển xác suất cao hơn. β lớn hơn (0,5) giữ policy gần SFT, nên margin nhỏ hơn và câu trả lời ít thay đổi hơn nữa so với β = 0,1 (vốn đã gần như không đổi). Vì lần chạy chính đã yếu, tôi dự đoán β = 0,05 là hướng đáng thử nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: dựng văn bản huấn luyện SFT thủ công (prompt + câu trả lời + `<|im_end|>`) thay vì để chat template của Qwen3 render cả lượt assistant, và kiểm tra đầu ra SFT trước khi tin vào DPO.**

1. *Phương án thay thế:* dùng `apply_chat_template` cho cả lượt assistant, đúng như notebook gốc.
2. *Vì sao đổi:* lần chạy đầu cho thấy cả 58/58 câu trả lời của SFT và DPO bắt đầu bằng các token `<tool_call>` thừa. Nguyên nhân: template của Qwen3 render lượt assistant thành `<think>\n\n</think>\n\n<câu trả lời>`, nhưng prompt lúc sinh chỉ kết thúc ở `assistant\n`, nên SFT dạy mô hình mở đầu bằng các token lạ. Sau khi sửa, lỗi thứ hai lộ ra: văn bản kết thúc bằng `<|im_end|>\n` mà SFTTrainer lại tự nối thêm EOS, nên mô hình học "xuống dòng → `<|im_end|>`" và trả về câu **rỗng** (22/58). Tôi loại trừ tokenizer, left padding, lượng tử hoá 4-bit, Unsloth inference và độ chính xác fp16/bf16 trước khi tìm ra điều này. Sửa: văn bản kết thúc đúng `<|im_end|>` không xuống dòng. Chạy lại sạch cho 0 câu rỗng và 0 `tool_call` ở cả hai mô hình.
3. *Kết quả:* làm tôi bất ngờ. Số liệu DPO lần đầu (accuracy 0,66, INTENDED) trông bình thường dù mô hình SFT đã hỏng, nên chỉ nhìn đường reward thì không phát hiện được; lỗi chỉ lộ ra khi đọc câu trả lời thật ở NB4. Một lỗi thứ ba cũng nhỏ nhưng đáng ghi: `import unsloth` ở đầu NB4 vá transformers làm reward model Qwen3 chấm sai (sanity 6/12, hội đồng rơi còn một giám khảo, thấy ở ô §3 của notebook). Chạy riêng trong một tiến trình không nạp unsloth thì đạt 12/12 (cả fp16 và bf16), nên tôi chấm lại bằng tiến trình riêng; ô chấm lại nằm ngay dưới §3.
4. *Làm lại thì đổi:* thêm một cổng kiểm tra ngay sau NB1 (giải mã vài câu từ mô hình đã gộp, đòi không rỗng và không có `tool_call`) trước khi chạy NB2–NB3, và chấm bằng reward model trong tiến trình riêng ngay từ đầu.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Không chạy. `lm-eval` chạy hơn 100 phút trên T4 rồi Colab hết compute units, nên không có số liệu để báo cáo.

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

---

## 8. Biến thể loss (bonus NB3b)

> Không chạy.

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

---

## 9. GRPO (bonus NB7)

> Không chạy.

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | |
| Sai số chuẩn ≈ √(p(1−p)/n) | |

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)

---

## Điều bất ngờ nhất

DPO ở NB3 trông "khoẻ" (INTENDED, accuracy 0,66) ngay cả khi mô hình SFT bên dưới đã hỏng: đường reward không thể hiện được lỗi chat template. Chỉ khi đọc câu trả lời thật (toàn `<tool_call>`, rồi toàn câu rỗng) mới thấy vấn đề.
