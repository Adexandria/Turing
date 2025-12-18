FROM python:3.12

# Create a non-root user to run the application and set permissions
RUN useradd -m -u 1000 turinguser
RUN mkdir -p /app/models && chown -R turinguser:turinguser /app /app/models
USER turinguser

# Set environment variables 
# PATH to include local user binaries and project root
ENV PATH="/home/turinguser/.local/bin:$PATH"
ENV PROJ_ROOT=/app

# Set the working directory in the container
WORKDIR /app

# Copy essential files to install dependencies
COPY --chown=turinguser requirements.txt .

# Install Python dependencies
RUN pip install --default-timeout=1000 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -v -r requirements.txt --upgrade --default-timeout=1000 --no-cache-dir --break-system-packages

# Copy remaining project files
COPY --chown=turinguser turing ./turing
COPY --chown=turinguser reports ./reports

# Expose port 7860 for the FastAPI application
EXPOSE 7860

# Default command to run the FastAPI application on port 7860
CMD ["uvicorn", "turing.api.app:app", "--host", "0.0.0.0", "--port", "7860"]