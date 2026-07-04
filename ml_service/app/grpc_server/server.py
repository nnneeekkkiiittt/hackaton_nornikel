from concurrent import futures
import grpc

from app.grpc_server.service import MLService
from app import ml_service_pb2_grpc


def serve():
    options = [
        ('grpc.max_receive_message_length', 64 * 1024 * 1024),
        ('grpc.max_send_message_length', 64 * 1024 * 1024)
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=options
    )

    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLService(),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("ML gRPC server started")

    server.wait_for_termination()