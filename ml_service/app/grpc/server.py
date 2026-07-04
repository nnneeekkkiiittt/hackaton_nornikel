from concurrent import futures
import grpc

from .service import MLService
import ml_service_pb2_grpc


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLService(),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("ML gRPC server started")

    server.wait_for_termination()