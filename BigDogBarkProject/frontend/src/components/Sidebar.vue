<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="logo-icon">
        <img src="@/assets/bigdog-bark-meme.png" alt="BigDog Bark" />
      </div>
      <div>
        <h2>大狗叫</h2>
        <p class="sidebar-subtitle">BigDog Bark</p>
      </div>
    </div>
    <nav class="sidebar-nav">
      <button @click="onNewChat" :class="['nav-btn', { active: chatStore.activeNav === 'newChat' }]">
        <i class="fas fa-bullhorn"></i> 放狗开搜
      </button>
      <button @click="onHistory" :class="['nav-btn', { active: chatStore.activeNav === 'history' }]">
        <i class="fas fa-clock-rotate-left"></i> 吠叫记录
      </button>
      <!-- 设置按钮已隐藏（Phase 1 不包含 RAG/文档管理） -->
    </nav>
    <div class="sidebar-footer">
      <div class="bark-card">
        <span class="bark-label">今日状态</span>
        <strong>大狗大狗叫叫叫</strong>
        <small>把路径丢过来，我开叫。</small>
      </div>
      <button @click="chatStore.handleClearChat" class="danger-btn">
        <i class="fas fa-broom"></i> 清空犬舍
      </button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

const chatStore = useChatStore();
const sessionStore = useSessionStore();

const onNewChat = () => {
  chatStore.handleNewChat();
};

const onHistory = async () => {
  chatStore.activeNav = 'history';
  sessionStore.showHistorySidebar = !sessionStore.showHistorySidebar;
  if (sessionStore.showHistorySidebar) {
    try {
      await sessionStore.fetchSessions();
    } catch (error: any) {
      alert(error.message);
    }
  }
};
</script>
