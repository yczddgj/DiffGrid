<script setup lang="ts">
import { ref } from 'vue';
import { useMainStore } from '../stores';

const store = useMainStore();
const isSubmitting = ref(false);
const modes = [
    {
        label: "DiffGrid",
        value: "default",
        description: "Regularized Multi-scale Difference Map"
    },
    {
        label: "Tabular",
        value: "tabular",
        description: "Regularized Multi-scale Difference Map"
    },
    {
        label: "Text",
        value: "text",
        description: "Regularized Multi-scale Difference Map"
    },
    {
        label: "KDE",
        value: "kde",
        description: "Kernel Density Estimation Difference"
    },
    {
        label: "LSDD",
        value: "lsdd",
        description: "Least-Squares Density Difference"
    }
];


const selectMode = async (mode: string) => {
    if (isSubmitting.value) return;
    try {
        isSubmitting.value = true;
        await store.setMode(mode);
        await store.initializeApp();
    } finally {
        isSubmitting.value = false;
    }
};
</script>

<template>
    <div class="flex items-center justify-center min-h-screen bg-gray-100 p-6">
        <!-- 外层容器 -->
        <div class="w-full max-w-4xl bg-white rounded-3xl shadow-md border border-gray-200 p-8">
            <!-- 标题 -->
            <h2 class="text-2xl font-bold text-gray-900 text-center mb-10">Select Mode</h2>

            <!-- 模式卡片容器 -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <button v-for="mode in modes" :key="mode.value" @click="selectMode(mode.value)" :disabled="isSubmitting"
                    class="group relative flex flex-col justify-center text-left rounded-2xl
                 border border-gray-200 bg-gray-50 p-8 shadow-lg
                 hover:bg-gray-100 hover:shadow-md hover:-translate-y-0.5
                 transition-all duration-200
                 focus:outline-none focus-visible:ring-2 disabled:opacity-50 disabled:cursor-not-allowed">
                    <!-- 模式标题 -->
                    <h3 class="text-lg font-bold text-gray-900 mb-2">
                        {{ mode.label }}
                    </h3>
                    <!-- 模式描述 -->
                    <p class="text-sm text-gray-600 leading-relaxed">
                        {{ mode.description }}
                    </p>
                </button>
            </div>
        </div>
    </div>
</template>
