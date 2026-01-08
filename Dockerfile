FROM python:3.11
ADD . .
RUN pip install requests aiohttp python-dotenv aidial-client
CMD ["python", "./run.py"] 