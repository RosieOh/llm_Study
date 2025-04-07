from django.shortcuts import render
from django.http import JsonResponse
from newspaper import Article
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os

# OpenAI API Key 설정
openai.api_key = 'sk-proj-1234567890'

# 모델 초기화
model_name = "kakaocorp/kanana-nano-2.1b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 뉴스 요약 함수
def summarize_news(url, system_prompt):
    try:
        # 뉴스 크롤링
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text[:200].replace("\n\n", ' ')

        # Kanana 모델
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        _ = model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=300,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = output.sequences[:, input_ids.size(1):]
        kanana_result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # GPT-4.5 모델
        completion = openai.ChatCompletion.create(
            model="gpt-4.5-preview-2025-02-27",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        gpt_result = completion.choices[0].message.content

        return title, kanana_result, gpt_result
    except Exception as e:
        return "", f"오류 발생: {e}", ""

# Django 뷰 생성
def compare_summaries(request):
    if request.method == "GET":
        return render(request, 'summary/home.html')  # 템플릿 페이지 표시

    elif request.method == "POST":
        url = request.POST.get("url")
        system_prompt = request.POST.get("system_prompt", '당신은 뉴스 기사를 영어로 번역하는 역할을 수행합니다. 주어진 뉴스 기사 내용만을 영어로 번역해주세요.')

        title, kanana_result, gpt_result = summarize_news(url, system_prompt)

        return JsonResponse({
            'title': title,
            'kanana_summary': kanana_result,
            'gpt_summary': gpt_result
        })
