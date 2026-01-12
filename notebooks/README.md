# SynthGen Notebook

> **Generate synthetic datasets using LLaMA 3.1 on Google Colab**

## Why I Built This

I got tired of manually creating fake data for my projects. You know the drill - you need 100 rows of customer data to test your dashboard, or some product records to demo your app, and you end up copy-pasting from random generators or writing tedious scripts that produce obviously fake-looking data.

So I thought: why not just tell an LLM what I want and let it figure out the rest?

That's basically what this notebook does. You describe your data in plain English - "give me 50 employees with names, departments, and realistic salaries" - and it generates a proper pandas DataFrame you can actually use.

## Why Transformers & HuggingFace?

I went with HuggingFace's `transformers` library because:

1. **Access to good models** - Meta's LLaMA 3.1 is genuinely impressive at following structured instructions, and HuggingFace makes it dead simple to load
2. **4-bit quantization** - Here's the thing: these models are huge. LLaMA 3.1 8B would normally need ~16GB of VRAM. But with `bitsandbytes` quantization, I got it running on Colab's free T4 GPU (~15GB). The quality drop is minimal, and it means anyone can run this without paying for cloud compute
3. **It just works** - The HuggingFace ecosystem handles all the annoying stuff (tokenization, chat templates, model loading) so I could focus on the actual data generation logic

## The Tricky Part

Getting LLMs to output valid JSON consistently is harder than it sounds. They love to add "helpful" explanations, forget commas, or generate arrays with mismatched lengths. I spent a good amount of time on the prompt engineering and added fallback parsing to handle the model's occasional creativity.

---

## Quick Start

1. **Open in Colab**: Click the badge below to open directly in Google Colab

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mSyOFFoEpMuvVx2uOd9NhnS7M1MqUl-s?usp=sharing)

2. **Set up HuggingFace Token**:
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) and create a token
   - In Colab, click the key icon in the left sidebar
   - Add a secret named `HF_TOKEN` with your token value

3. **Enable GPU Runtime**:
   - Go to `Runtime` → `Change runtime type`
   - Select `T4 GPU` (or better)

4. **Run All Cells**: Execute all cells in order

## Features

| Feature | Description |
|---------|-------------|
| **Natural Language Input** | Describe your data in plain English |
| **Structured Output** | Get pandas-compatible DataFrames |
| **4-bit Quantization** | Efficient memory usage on free Colab GPUs |
| **Streamlit UI** | Interactive web interface for data generation |
| **CSV Export** | Download your generated datasets |

## Example Prompts

### Customer Data
```
Customer data with:
- first_name and last_name
- age between 25 and 55
- email addresses
- city (US cities)
```

### Product Catalog
```
E-commerce products:
- product_name (electronics)
- price between $10 and $500
- category (phones, laptops, accessories)
- in_stock (boolean)
- rating (1-5 stars)
```

### Employee Records
```
Employee data:
- name (first and last)
- department (Engineering, Sales, Marketing, HR)
- salary ($40,000 - $150,000)
- years_employed (0-20)
- is_manager (boolean, ~10% true)
```

### Transaction Data
```
Bank transactions:
- transaction_id (unique alphanumeric)
- amount ($5 - $500)
- category (groceries, electronics, dining, utilities)
- date (last 30 days)
- is_fraud (boolean, ~2% true)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rows` | 10 | Number of rows to generate (1-100) |
| `temperature` | 0.7 | Creativity level (0.1-1.5) |
| `MODEL_ID` | LLaMA 3.1 8B | HuggingFace model identifier |

### Temperature Guide
- **0.1-0.3**: Very consistent, predictable data
- **0.4-0.6**: Balanced consistency and variety
- **0.7-0.9**: More creative, diverse outputs
- **1.0+**: Highly creative, may be less structured

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  User Prompt    │────▶│  LLaMA 3.1   │────▶│   JSON      │
│  (Natural Lang) │     │  (4-bit)     │     │   Output    │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                        ┌──────────────┐             │
                        │   pandas     │◀────────────┘
                        │  DataFrame   │
                        └──────────────┘
```

## Requirements

- **Google Colab** with GPU runtime (T4 recommended)
- **HuggingFace Account** with LLaMA model access
- **~8GB GPU RAM** (provided by Colab T4)

### Package Dependencies
- `torch` (CUDA 12.4)
- `transformers`
- `bitsandbytes`
- `accelerate`
- `streamlit`
- `pyngrok`

## 🔧 Troubleshooting

### "CUDA out of memory"
- Restart runtime and try with fewer rows
- Reduce `max_new_tokens` in generation config

### "Failed to parse JSON"
- Simplify your prompt
- Lower the temperature setting
- Be more specific about column names

### "Model not found"
- Ensure you have access to LLaMA models on HuggingFace
- Accept the model license agreement
- Verify your HF_TOKEN is correct

### Streamlit not loading
- Wait a few seconds and refresh
- Check if ngrok URL is accessible
- Try rerunning the Streamlit cell

## 📄 License

This notebook is provided for educational and research purposes. Please ensure compliance with:
- [Meta LLaMA License](https://llama.meta.com/llama3/license/)
- [HuggingFace Terms of Service](https://huggingface.co/terms-of-service)

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
1. Open an issue
2. Submit a pull request
3. Share your creative use cases!

---

**Made with ❤️ using LLaMA 3.1 and HuggingFace Transformers**

