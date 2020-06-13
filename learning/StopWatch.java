
import java.util.Hashtable;

/**
 *  * A simple Stop Watch Implementation. It can be used for various tasks.
 *   * 
 *    * @author developerfeed
 *     * 
 *      */
class StopWatch {
    private static final Hashtable<String, Long> tasks = new Hashtable<String, Long>();

    /**
     *   * Start the Watch for a Task with Id
     *       * 
     *           * @param id
     *               */
    public static void start(String taskId) {
        tasks.put(taskId, new Long(System.currentTimeMillis()));

    }

    /**
     *   * Stop the watch
     *       * 
     *           * @param id
     *               * @return
     *                   */
    public static long stop(String taskId) {
        return System.currentTimeMillis()
            - ((Long) tasks.remove(taskId)).longValue();

    }

    /**
     *   * Dummy Task 1
     *       */
    public static void executeTask1() {
        System.out.println("Task 1 done");

    }

    /**
     *   * Dummy Task 2
     *       */
    public static void executeTask2() {
        System.out.println("Task 2 done");

    }

    /**
     *   * Main Test Method
     *       * 
     *           * @param args
     *               */
    public static void main(String[] args) {
        // Start a global stopwatch
        StopWatch.start("GLOBAL");

        // evaluate time used by task 1
        StopWatch.start("TASK1");
        executeTask1();
        System.out.println("Time elapsed for task 1 : "
                + StopWatch.stop("TASK1") + "ms");

        // evaluate time used by task 2
        StopWatch.start("TASK2");
        executeTask2();
        System.out.println("Time elapsed for task 2 : "
                + StopWatch.stop("TASK2") + "ms");

        // Display time elapsed for full processing
        System.out.println("Total processing time : "
                + StopWatch.stop("GLOBAL") + "ms");

    }

}
