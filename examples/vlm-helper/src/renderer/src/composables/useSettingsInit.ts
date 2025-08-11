import { onMounted, onUnmounted } from 'vue'
import { useSettingsStore } from '../stores/settingsStore'

/**
 * 用于初始化设置的组合函数
 * 在应用启动时调用，确保设置正确加载和同步
 */
export function useSettingsInit(): { settingsStore: ReturnType<typeof useSettingsStore> } {
  const settingsStore = useSettingsStore()
  let cleanup: (() => void) | null = null

  onMounted(() => {
    // 加载设置
    settingsStore.loadSettings()

    // 设置跨窗口同步
    cleanup = settingsStore.setupCrossWindowSync()

    console.log('设置初始化完成')
  })

  onUnmounted(() => {
    // 清理跨窗口同步监听器
    if (cleanup) {
      cleanup()
    }
  })

  return {
    settingsStore
  }
}
