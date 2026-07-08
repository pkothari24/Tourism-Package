# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set up a secure, non-root user for Hugging Face Spaces execution environment
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Establish application working directory inside user home context
WORKDIR $HOME/app

# Copy application files and grant permissions to the user container thread directly
COPY --chown=user . $HOME/app

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir --user -r requirements.txt

# EXPOSE port 7860 for Hugging Face Spaces infrastructure compatibility
EXPOSE 7860

# Define the command to run the Streamlit app on port "7860" 
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
