import os
import torch
import pandas as pd
import gc
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from huggingface_hub import scan_cache_dir

# ============================================================
# 1. DANH SÁCH 10 MODEL (Đã fix tên chuẩn)
# ============================================================
model_candidates = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct", # Lưu ý: Model này cần triton, có thể vẫn fail trên Windows
    "olmo-2-1b": "allenai/OLMo-2-0425-1B-SFT",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "vaultgemma-1b": "google/vaultgemma-1b",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3", 
}

use_4bit = True

# ============================================================
# 2. HÀM XÓA CACHE Ổ CỨNG (QUAN TRỌNG)
# ============================================================
def delete_model_from_disk(repo_id):
    """
    Tìm và xóa model cụ thể trong cache HuggingFace để giải phóng ổ cứng.
    """
    print(f"   [DISK CLEANUP] Dang quet cache de xoa: {repo_id}...")
    try:
        hf_cache_info = scan_cache_dir()
        deleted_size = 0
        
        for repo in hf_cache_info.repos:
            if repo.repo_id == repo_id:
                # Tìm thấy repo, thực hiện xóa
                folder_path = str(repo.repo_path)
                print(f"   -> Tim thay folder: {folder_path}")
                
                # Cách 1: Dùng API delete của huggingface (an toàn)
                delete_strategy = hf_cache_info.delete_revisions(*repo.revisions)
                delete_strategy.execute()
                
                print(f"   -> DA XOA THANH CONG khoi o cung!")
                return
        
        print("   -> Khong tim thay model trong cache (co the chua tai xong hoac da xoa).")
        
    except Exception as e:
        print(f"   !!! LOI KHI XOA DISK: {e}")
        print("   Hay xoa thu cong bang lenh: huggingface-cli delete-cache")

# ============================================================
# 3. HÀM CHECK MODEL
# ============================================================
def check_one_model(model_key, model_id):
    print(f"\n{'='*60}")
    print(f"Checking: {model_key} ({model_id})")
    
    result = {"model_key": model_key, "status": "FAILED", "note": ""}

    try:
        # Tải Config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Tải Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Cấu hình 4-bit
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        
        # Tải Model (Có bật CPU Offload cho máy yếu)
        print("... Dang tai va load model (co the lau)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True # Quan trọng cho máy yếu
        )

        result["status"] = "SUCCESS"
        print(f"OK! {model_key} chay duoc tren may nay.")

    except Exception as e:
        print(f"!!! ERROR: {str(e)}")
        result["note"] = str(e)[0:150] # Lấy đoạn lỗi ngắn
    
    finally:
        # BƯỚC 1: Xóa khỏi RAM
        print("... Don dep RAM/VRAM ...")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        # BƯỚC 2: Xóa khỏi Ổ CỨNG (Disk)
        delete_model_from_disk(model_id)
        
    return result

# ============================================================
# 4. CHẠY CHƯƠNG TRÌNH
# ============================================================
def main():
    print("### KIEM TRA MODEL & TU DONG DON DEP O CUNG ###")
    print("Luu y: Mistral 7B rat nang (15GB), hay dam bao o C con trong it nhat 20GB truoc khi chay script nay.")
    
    summary = []
    
    # Duyệt qua từng model
    for key, mid in model_candidates.items():
        res = check_one_model(key, mid)
        summary.append(res)
        
        # In nhanh kết quả sau mỗi lần chạy
        print(f"ket qua tam thoi: {key} -> {res['status']}")

    print("\n=== TONG KET CUOI CUNG ===")
    df = pd.DataFrame(summary)
    print(df)
    df.to_csv("final_model_check.csv", index=False)

if __name__ == "__main__":
    main()