
class Timer {
  private val startTime: Long = System.currentTimeMillis

  def elapsed = (System.currentTimeMillis - startTime).toFloat / 1000
}
