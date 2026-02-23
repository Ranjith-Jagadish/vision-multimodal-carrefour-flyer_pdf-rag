# Testing Oreo Query with Sample PDF

This guide shows you how to test the RAG system with the sample Carrefour PDF to query about Oreo offers.

## Quick Test

### Option 1: Using the Test Script

```bash
cd pdf-rag-system
source venv/bin/activate
python test_oreo_query.py
```

This script will:
1. Process the sample PDF
2. Add it to the vector store
3. Query about Oreo
4. Display results with citations and verify images are generated

### Option 2: Using the Web Interface

1. **Start the backend:**
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Start the frontend:**
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

3. **Upload the PDF:**
   - Open http://localhost:8501
   - In the sidebar, upload: `sample_pdf/6 - 15 Jan Super Deals Digital Leaflet.pdf`
   - Wait for processing to complete

4. **Query about Oreo:**
   - In the chat interface, type: "What is the offer on Oreo?"
   - Or: "Tell me about Oreo biscuits"
   - Or: "What's the price of Oreo?"

5. **View Citations:**
   - Expand the "Citations" section
   - You should see:
     - Page number where Oreo is mentioned
     - Text snippet from that page
     - **Image of the PDF page** showing the Oreo product

### Expected Results

When querying about Oreo, you should get:

**Answer:**
- Information about Oreo original biscuit
- Price: 17.49 (discounted from 36.00)
- Discount: -51%
- Quantity: 24 pieces, 36.8g each

**Citations:**
- Page number where Oreo appears
- Text snippet containing "Oreo original biscuit 36.8g 24 pieces"
- **Visual citation**: Image of the PDF page showing the Oreo product with price tag

## Verification Checklist

✅ PDF is processed and chunks are extracted
✅ Query returns relevant information about Oreo
✅ Citations include page numbers
✅ Citation images are generated and displayed
✅ Image shows the actual PDF page with Oreo product

## Troubleshooting

### Citation images not showing

1. **Check Poppler installation:**
   ```bash
   # macOS
   brew install poppler
   
   # Verify
   pdftoppm -h
   ```

2. **Check image directory:**
   ```bash
   ls -la static/citations/
   ```
   Should contain PNG files like `{hash}_page_{number}.png`

3. **Check PDF path in metadata:**
   - Ensure the PDF path stored in vector store metadata is correct
   - Should be absolute path to the uploaded PDF

### No results for Oreo query

1. **Check if PDF was processed:**
   ```bash
   # List documents
   curl http://localhost:8000/api/documents
   ```

2. **Check vector store:**
   - Verify chunks were added
   - Check if embedding model loaded correctly

3. **Try different queries:**
   - "Oreo"
   - "biscuit"
   - "Oreo original"
   - "24 pieces biscuit"

## API Test (Alternative)

You can also test via API:

```bash
# Upload PDF
curl -X POST http://localhost:8000/api/upload \
  -F "file=@sample_pdf/6 - 15 Jan Super Deals Digital Leaflet.pdf"

# Query about Oreo
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the offer on Oreo?",
    "top_k": 3
  }'
```

The response will include:
- `answer`: Generated answer
- `citations`: Array with page numbers, text snippets, and `image_url`
- Access citation images at: `http://localhost:8000{image_url}`

