import sys
import socket
import driver
import xml.etree.ElementTree as ET
import os

SCR_SERVER_PATH = r"C:\Users\ayaan\Documents\University\Semester 6\Aritificial Intelligence\Project\torcs\drivers\scr_server\scr_server.xml"
NEW_CAR_NAME = "p406"  # Change this to the desired car model

def update_car_model():
    try:
        tree = ET.parse(SCR_SERVER_PATH)
        root = tree.getroot()
        
        for section in root.findall(".//section[@name='index']/section"):
            car = section.find("attstr[@name='car name']")
            if car is not None:
                car.set("val", NEW_CAR_NAME)
        
        tree.write(SCR_SERVER_PATH)
        print(f"Updated all car models to {NEW_CAR_NAME} in scr_server.xml")
    except Exception as e:
        print(f"Error updating car model: {e}")

def main():
    update_car_model()  # Update the car before starting the client

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