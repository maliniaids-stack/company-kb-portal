FROM python:3.11-slim

# Set up a working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code
COPY . .

# CRITICAL: Hugging Face Spaces requires port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]