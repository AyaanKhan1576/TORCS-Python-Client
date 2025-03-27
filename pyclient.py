import sys
import argparse
import socket
import driver

def main():
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

    parser.add_argument('--host', dest='host_ip', default='localhost', help='Host IP address (default: localhost)')
    parser.add_argument('--port', type=int, dest='host_port', default=3001, help='Host port number (default: 3001)')
    parser.add_argument('--id', dest='id', default='SCR', help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', dest='max_episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--maxSteps', dest='max_steps', type=int, default=0, help='Max steps (0 = unlimited)')
    parser.add_argument('--track', dest='track', default=None, help='Track name')
    parser.add_argument('--stage', dest='stage', type=int, default=0, help='Stage (0=Warm-Up, 1=Qualifying, 2=Race, 3=Unknown)')

    args = parser.parse_args()

    print(f'Connecting to server {args.host_ip}:{args.host_port}')
    print(f'Bot ID: {args.id} | Episodes: {args.max_episodes} | Max Steps: {args.max_steps} | Track: {args.track} | Stage: {args.stage}')
    print('*********************************************')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error as e:
        print(f'Socket error: {e}')
        sys.exit(-1)

    d = driver.Driver(args.stage)
    shutdown = False
    episode = 0

    while not shutdown:
        while True:
            print(f'[Client] Sending ID to server: {args.id}')
            init_msg = args.id + d.init()
            print(f'[Client] Init message: {init_msg}')

            try:
                sock.sendto(init_msg.encode(), (args.host_ip, args.host_port))
                data, _ = sock.recvfrom(1000)
                data = data.decode()
            except socket.timeout:
                print("[Client] No response from server (timeout). Retrying...")
                continue
            except socket.error as e:
                print(f"[Client] Socket error: {e}")
                sys.exit(-1)

            if '***identified***' in data:
                print('[Client] Received:', data)
                break

        step = 0
        while True:
            try:
                buf, _ = sock.recvfrom(1000)
                buf = buf.decode()
            except socket.timeout:
                print("[Client] No response from server during episode.")
                continue
            except socket.error as e:
                print(f"[Client] Socket error: {e}")
                sys.exit(-1)

            if '***shutdown***' in buf:
                d.onShutDown()
                shutdown = True
                print('[Client] Shutdown signal received.')
                break

            if '***restart***' in buf:
                d.onRestart()
                print('[Client] Restart signal received.')
                break

            step += 1
            if args.max_steps == 0 or step < args.max_steps:
                action = d.drive(buf)
            else:
                action = '(meta 1)'

            try:
                sock.sendto(action.encode(), (args.host_ip, args.host_port))
            except socket.error as e:
                print(f"[Client] Failed to send action: {e}")
                sys.exit(-1)

        episode += 1
        if episode >= args.max_episodes:
            shutdown = True

    sock.close()

if __name__ == '__main__':
    main()
