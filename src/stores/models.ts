import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ModelInfo, ModelFormat } from '@types/index'

export const useModelsStore = defineStore('models', () => {
  const models = ref<ModelInfo[]>([])
  const isLoading = ref(false)
  const selectedModelId = ref<string | null>(null)

  const downloadedModels = computed(() => models.value.filter(m => m.downloaded))
  const availableModels = computed(() => models.value.filter(m => !m.downloaded))
  const selectedModel = computed(() => models.value.find(m => m.id === selectedModelId.value))

  async function loadModels() {
    isLoading.value = true
    try {
      // Try to sync with Modelscope API first
      await syncWithModelscope('llm gguf')
      
      // If no models found, use mock data as fallback
      if (models.value.length === 0) {
        const mockModels: ModelInfo[] = [
          {
            id: 'qwen2.5-7b-instruct-q4',
            name: 'Qwen2.5-7B-Instruct-Q4',
            description: 'Qwen 2.5 7B Instruct model with 4-bit quantization',
            size: 4.2,
            format: ModelFormat.GGUF,
            quantization: 'Q4_K_M',
            contextLength: 32768,
            tags: ['qwen', 'instruct', 'chat', 'multilingual'],
            downloadUrl: 'https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf',
            downloaded: false
          },
          {
            id: 'llama3.1-8b-instruct-q6',
            name: 'Llama 3.1 8B Instruct Q6',
            description: 'Llama 3.1 8B Instruct model with 6-bit quantization',
            size: 6.8,
            format: ModelFormat.GGUF,
            quantization: 'Q6_K',
            contextLength: 128000,
            tags: ['llama', 'instruct', 'chat', 'english'],
            downloadUrl: 'https://www.modelscope.cn/models/meta-llama/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct-q6_k.gguf',
            downloaded: false
          },
          {
            id: 'deepseek-coder-7b-q5',
            name: 'DeepSeek Coder 7B Q5',
            description: 'DeepSeek Coder 7B model optimized for coding tasks',
            size: 5.1,
            format: ModelFormat.GGUF,
            quantization: 'Q5_K_M',
            contextLength: 16384,
            tags: ['deepseek', 'coder', 'programming', 'code'],
            downloadUrl: 'https://www.modelscope.cn/models/deepseek-ai/DeepSeek-Coder-7B-GGUF/resolve/main/deepseek-coder-7b-q5_k_m.gguf',
            downloaded: true,
            path: '/models/deepseek-coder-7b-q5_k_m.gguf'
          },
          {
            id: 'yi-6b-chat-q4',
            name: 'Yi 6B Chat Q4',
            description: 'Yi 6B Chat model with 4-bit quantization',
            size: 3.8,
            format: ModelFormat.GGUF,
            quantization: 'Q4_K_M',
            contextLength: 4096,
            tags: ['yi', 'chat', 'chinese', 'english'],
            downloadUrl: 'https://www.modelscope.cn/models/01-ai/Yi-6B-Chat-GGUF/resolve/main/yi-6b-chat-q4_k_m.gguf',
            downloaded: false
          },
          {
            id: 'baichuan2-7b-chat-q5',
            name: 'Baichuan2 7B Chat Q5',
            description: 'Baichuan2 7B Chat model with 5-bit quantization',
            size: 4.9,
            format: ModelFormat.GGUF,
            quantization: 'Q5_K_M',
            contextLength: 4096,
            tags: ['baichuan', 'chat', 'chinese', 'bilingual'],
            downloadUrl: 'https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-GGUF/resolve/main/baichuan2-7b-chat-q5_k_m.gguf',
            downloaded: false
          }
        ]
        
        models.value = mockModels
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    } finally {
      isLoading.value = false
    }
  }

  async function syncWithModelscope(query: string = 'llm gguf') {
    try {
      const response = await fetch(`/api/v1/models/sync?query=${encodeURIComponent(query)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const result = await response.json()
      if (result.success && result.data.new_models.length > 0) {
        // Add new models to the store
        const newModels = result.data.new_models.map((modelData: any) => ({
          id: modelData.id,
          name: modelData.name,
          description: modelData.description,
          size: modelData.size,
          format: modelData.format.toUpperCase() as ModelFormat,
          quantization: modelData.quantization,
          contextLength: modelData.context_length,
          tags: modelData.tags,
          downloadUrl: modelData.download_url,
          downloaded: modelData.downloaded,
          path: modelData.path
        }))
        
        // Merge with existing models
        const existingIds = new Set(models.value.map(m => m.id))
        const trulyNewModels = newModels.filter((m: ModelInfo) => !existingIds.has(m.id))
        
        if (trulyNewModels.length > 0) {
          models.value = [...models.value, ...trulyNewModels]
          console.log(`Synced ${trulyNewModels.length} new models from Modelscope`)
        }
      }
    } catch (error) {
      console.warn('Failed to sync with Modelscope API, using local models:', error)
    }
  }

  function selectModel(modelId: string) {
    selectedModelId.value = modelId
  }

  function addModel(model: ModelInfo) {
    const existingIndex = models.value.findIndex(m => m.id === model.id)
    if (existingIndex >= 0) {
      models.value[existingIndex] = model
    } else {
      models.value.push(model)
    }
  }

  function updateModel(modelId: string, updates: Partial<ModelInfo>) {
    const model = models.value.find(m => m.id === modelId)
    if (model) {
      Object.assign(model, updates)
    }
  }

  function removeModel(modelId: string) {
    const index = models.value.findIndex(m => m.id === modelId)
    if (index >= 0) {
      models.value.splice(index, 1)
    }
  }

  function markAsDownloaded(modelId: string, path: string) {
    updateModel(modelId, { downloaded: true, path })
  }

  function searchModels(query: string) {
    if (!query.trim()) {
      return models.value
    }
    
    const lowercaseQuery = query.toLowerCase()
    return models.value.filter(model => 
      model.name.toLowerCase().includes(lowercaseQuery) ||
      model.description.toLowerCase().includes(lowercaseQuery) ||
      model.tags.some(tag => tag.toLowerCase().includes(lowercaseQuery))
    )
  }

  function filterByFormat(format: ModelFormat) {
    return models.value.filter(model => model.format === format)
  }

  function filterByTag(tag: string) {
    return models.value.filter(model => model.tags.includes(tag))
  }

  function updateModelsList(newModels: ModelInfo[]) {
    models.value = newModels
  }

  return {
    models,
    isLoading,
    selectedModelId,
    downloadedModels,
    availableModels,
    selectedModel,
    loadModels,
    selectModel,
    addModel,
    updateModel,
    removeModel,
    markAsDownloaded,
    searchModels,
    filterByFormat,
    filterByTag,
    updateModelsList
  }
})