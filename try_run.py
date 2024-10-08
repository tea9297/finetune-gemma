import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datetime

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)

model_id = "model/gemma2-27b-panaSQL-Instruct-Finetune-merged_model"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda:0"
    cur_datetime = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    <start_of_turn>user
    "You are a powerful text-to-SQL model. Your job is to answer questions about a SQLite database. 現在有一個SQLite的表格sales_result，其欄位如下:\n\nYYYYMM: 銷售年月，資料刻度為\"月\"\n課別中文名稱: 業務課別的中文名稱，如電池零件一課\n客戶簡稱: 客戶公司的中文名稱，如廣達電腦、富士康科技\n產品PRODUCT: 產品Group的名稱，若問產品的銷售實績/數量，以此欄位為主\n業務員名稱: 編碼為2-4個中文字或是純英文,\n業界: 銷售對象所屬市場Market,\n銷售數量: 正或負值，近義詞包含\"銷貨量/出貨量\"，跨月份/年份的總和計算可能需要使用sum語法,\n銷售金額:正或負值(單位：台幣)，近義詞包含\"營業額/業績/銷售實績/銷售額/下單\"，跨月份/年份的總和計算可能需要使用sum語法,\n\n請根據以上表單的資訊，產生一個SQL語句，用於查詢對應使用者問題的SQLite表格資料。\n若使用者問題包含時間，根據當前日期{cur_datetime}推算出問題的絕對日期範圍;\n若詢問年份總和，請使用sum語法，或是substr(YYYYMM, 1, 4) AS sales_year。"
    {query}
    <end_of_turn>\n<start_of_turn>model

    """
    #prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs,
                                   max_new_tokens=1000,
                                   do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    # decoded = tokenizer.batch_decode(generated_ids)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return (decoded)


result = get_completion(query="列出前十大年度銷售金額越來越多的客戶",
                        model=model,
                        tokenizer=tokenizer)

print(result)
