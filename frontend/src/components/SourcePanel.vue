<script setup>
defineProps({
  sources: { type: Array, default: () => [] },
  hasQuestion: { type: Boolean, default: false },
  loading: { type: Boolean, default: false },
})
</script>

<template>
  <section class="source-section">
    <header class="panel-heading">
      <div><span class="eyebrow">RETRIEVAL CONTEXT</span><h2>检索来源</h2></div>
      <span v-if="sources.length" class="source-total">{{ sources.length }}</span>
    </header>
    <div v-if="sources.length" class="source-list">
      <article v-for="(source, index) in sources" :key="`${source.chunk_id}-${index}`" class="source-card">
        <div class="source-card-top"><span class="source-index">0{{ index + 1 }}</span><span class="match-badge">MATCH</span></div>
        <strong>{{ source.source }}</strong><span class="chunk-id">{{ source.chunk_id }}</span><p>{{ source.text }}</p>
      </article>
    </div>
    <div v-else class="source-empty">
      <div class="empty-orbit" aria-hidden="true"><span></span></div>
      <strong>{{ loading ? '正在整理依据' : hasQuestion ? '暂无可展示的知识来源' : '等待提问' }}</strong>
      <p>{{ loading ? '本次可用的知识来源会在回答完成后显示。' : hasQuestion ? '本次没有可展示的知识来源，可查看助手回复中的说明。' : '发送问题后，相关知识片段会显示在这里。' }}</p>
    </div>
  </section>
</template>
