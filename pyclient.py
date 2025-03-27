import sys
import socket
import driver

def main():
    # Default parameters
    host_ip = "localhost"
    host_port = 3001
    bot_id = "SCR"
    max_episodes = 1
    max_steps = 0
    track = None
    stage = 0

    print(f'Connecting to server {host_ip}:{host_port}')
    print(f'Bot ID: {bot_id} | Episodes: {max_episodes} | Max Steps: {max_steps} | Track: {track} | Stage: {stage}')
    print('*********************************************')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error as e:
        print(f'Socket error: {e}')
        sys.exit(-1)

    d = driver.Driver(stage)
    shutdown = False
    episode = 0

    while not shutdown:
        while True:
            print(f'[Client] Sending ID to server: {bot_id}')
            init_msg = bot_id + d.init()
            print(f'[Client] Init message: {init_msg}')

            try:
                sock.sendto(init_msg.encode(), (host_ip, host_port))
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
            if max_steps == 0 or step < max_steps:
                action = d.drive(buf)
            else:
                action = '(meta 1)'

            try:
                sock.sendto(action.encode(), (host_ip, host_port))
            except socket.error as e:
                print(f"[Client] Failed to send action: {e}")
                sys.exit(-1)

        episode += 1
        if episode >= max_episodes:
            shutdown = True

    sock.close()

if __name__ == '__main__':
    main()
