package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
)

// checks and handles error messages
func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// sends messages from command line to server, and prints response
// stops when exit message is typed
func msgConn(conn net.Conn) {
	for {
		var msg strings.Builder
		scanner := bufio.NewScanner(os.Stdin)
		if scanner.Scan() {
			msg.WriteString(scanner.Text())
		}
		check(scanner.Err())

		msg.WriteString("\000")
		_, err := conn.Write([]byte(msg.String()))
		check(err)

		buffer := make([]byte, 1024)
		_, err = conn.Read(buffer)
		check(err)
		fmt.Println(string(buffer))

		valid := map[string]bool{"exit\000": true, "quit\000": true, "stop\000": true}
		if valid[strings.ToLower(msg.String())] {
			break
		}
	}
}

func main() {
	// Connects to server over tcp
	addr, err := net.ResolveTCPAddr("tcp", ":34933")
	check(err)
	conn, err := net.DialTCP("tcp4", nil, addr)
	check(err)

	// Recieves and prints welcome message
	buffer := make([]byte, 1024)
	_, err = conn.Read(buffer)
	check(err)
	fmt.Println(string(buffer))

	msgConn(conn)

	listener, err := net.Listen("tcp", ":8080")
	check(err)
	defer listener.Close()

	_, err = conn.Write([]byte("Connect to: 10.0.0.115:8080\000"))
	check(err)

	buffer = make([]byte, 1024)
	_, err = conn.Read(buffer)
	check(err)
	fmt.Println(string(buffer))

	conn2, err := listener.Accept()
	check(err)

	buffer = make([]byte, 1024)
	_, err = conn2.Read(buffer)
	check(err)
	fmt.Println(string(buffer))

	msgConn(conn2)

	conn.Close()
	conn2.Close()

}
