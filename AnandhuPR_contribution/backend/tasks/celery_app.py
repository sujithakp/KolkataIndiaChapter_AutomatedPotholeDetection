from celery import Celery

# # Create Celery instance
# celery_app = Celery(
#     'pothole_detection',
#     broker='redis://localhost:6379/0',
#     backend='redis://localhost:6379/0'
# )

# # for redis in docker

celery_app = Celery(
    'pothole_detection',
    broker='redis://redis:6379/0',  # Use the service name 'redis' instead of localhost
    backend='redis://redis:6379/0'  # Same for backend
)



import tasks.video_tasks

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)