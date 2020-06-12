import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.Socket;

public class Client {
	// The remote process will run on localhost and listen on
	// port 65432.
	public static final String REMOTE_HOST = "";
	public static final int REMOTE_PORT = 65432;

	public static void main(String[] args) throws IOException {
		// Make a TCP connection to the remote process.
		Socket socket = new Socket(REMOTE_HOST, REMOTE_PORT);

		// Build BufferedWriter and BufferedReader from the socket so we
		// can do two-way text-based I/O.
		BufferedWriter sockOut = new BufferedWriter(
				new OutputStreamWriter(socket.getOutputStream()));
		BufferedReader sockIn = new BufferedReader(
				new InputStreamReader(socket.getInputStream()));

		// Build a BufferedReader around System.in so we can easily
		// read lines of input from the console.
		BufferedReader consoleIn = new BufferedReader(
				new InputStreamReader(System.in));

		// Read input from the console, send it to the remote process,
		// read the response, print it to the console, and repeat.
		for (;;) {
			// Get user input from the console.
			System.out.println("Type something to send to the server:");
			String line = consoleIn.readLine();
			if (line == null) {
				// No more input from the console.  We're done.
				break;
			}

			// Write the text to the remote process.
			sockOut.write(line, 0, line.length());
			sockOut.newLine();
			sockOut.flush();

			// Read the response from the remote process.
			String response = sockIn.readLine();
			if (response == null) {
				System.out.println("Remote process closed the connection.");
				break;
			}

			// Print the response to the console.
			System.out.println("Got back this response:");
			System.out.println(response);
			System.out.println();
		}
	}
}

