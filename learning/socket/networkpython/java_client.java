 import java.io.IOException;
 import java.io.InputStream;
 import java.io.OutputStream;
 import java.net.Socket;
 import java.io.PrintWriter;
 import java.io.BufferedReader;
 import java.io.InputStreamReader;

public class java_client {
    static String hostName = "HP-ENVY-Laptop-13-aq0xxx";
    static int portNumber = 12345;
    static int tam;

    public static void main(String[] args) throws IOException, InterruptedException {
        try {
            Socket echoSocket = new Socket(hostName, portNumber);
            //PrintWriter sout =
                //new PrintWriter(echoSocket.getOutputStream(), true);
            OutputStream sout = echoSocket.getOutputStream();
            BufferedReader in =
                new BufferedReader(
                    new InputStreamReader(echoSocket.getInputStream()));
                    
            char[] tmp = "abdasdc".toCharArray();
            for (char x :tmp){
                sout.write((int) x);
            }

            String userInput= "";
            for (int i =0; i <2; i++){
                int ch;
                ch = in.read();
                userInput = userInput + (char)ch; 
            }
            System.out.println("echo: " + userInput);
            
            echoSocket.close();
            //System.out.println(tmp);
        } catch (IOException ie){
            System.out.println("Can't connect to server");
        }
    }
}