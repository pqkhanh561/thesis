public class SavePaint extends JPanel
{

        public SavePaint()
        {
                    JFrame frame = new JFrame("TheFrame");
                            frame.add(this);
                                    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                                            frame.setSize(400,400);
                                                    frame.setVisible(true);

                                                            try
                                                            {
                                                                            BufferedImage image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_RGB);
                                                                                        Graphics2D graphics2D = image.createGraphics();
                                                                                                    frame.paint(graphics2D);
                                                                                                                ImageIO.write(image,"jpeg", new File("/home/deniz/Desktop/jmemPractice.jpeg"));
                                                                                                                        
                                                            }
                                                                    catch(Exception exception)
                                                                    {
                                                                                    //code
                                                                                    //        
                                                                    }
                                                                        
        }

            protected void paintComponent(Graphics g)
            {
                        g.drawRect(50,50,50,50);
                            
            }

                public static void main(String[] args)
                {
                            new SavePaint();
                                
                }
                
}
