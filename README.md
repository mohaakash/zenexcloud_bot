# ZenexCloud AI Chatbot

This is the backend API for the ZenexCloud AI Chatbot, a customer service AI designed to answer questions about ZenexCloud's services. This chatbot is built with Django and uses a knowledge base to provide answers.

## Features

*   **REST API:** A simple and easy-to-use REST API for interacting with the chatbot.
*   **Knowledge Base:** The chatbot uses a knowledge base in JSON format to answer questions.
*   **Vector Store:** The chatbot uses a vector store to find the most relevant answers from the knowledge base.

## Technologies Used

*   **Python:** The core programming language for the backend.
*   **Django:** A high-level Python web framework for building the REST API.
*   **Django REST Framework:** A powerful and flexible toolkit for building Web APIs in Django.
*   **LangChain:** A framework for developing applications powered by language models.
*   **FAISS:** A library for efficient similarity search and clustering of dense vectors.
*   **Hugging Face Transformers:** A library for state-of-the-art machine learning for PyTorch, TensorFlow, and JAX.

## Project Structure

```
.
├── chatbot_api/              # Django project
│   ├── api/                  # Django app for the chatbot API
│   │   ├── chatbot.py        # Chatbot logic
│   │   ├── urls.py           # API URL configuration
│   │   └── views.py          # API views
│   └── chatbot_api/          # Django project settings
├── files/                    # Directory for knowledge base files
├── knowledge_base.json       # Knowledge base in JSON format
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/zenexcloud_bot.git
    cd zenexcloud_bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Django development server:**
    ```bash
    python chatbot_api/manage.py runserver
    ```

The API will be available at `http://127.0.0.1:8000/`.

## API Endpoint

*   **URL:** `http://127.0.0.1:8000/api/chat/`
*   **Method:** `POST`
*   **Body:**
    ```json
    {
        "message": "Your message here"
    }
    ```

### Example with cURL

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
-H "Content-Type: application/json" \
-d '{"message": "What is ZenexCloud?"}'
```

You should receive a response like this:

```json
{
    "response": "ZenexCloud is a cloud computing platform that provides a wide range of services, including virtual machines, storage, and networking."
}
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.